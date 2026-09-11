package handlers

import (
	"context"
	"path/filepath"
	"slices"
	"sort"
	"testing"
	"time"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	configstoreTables "github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/grant"
	"github.com/maximhq/bifrost/framework/modelcatalog"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
)

func newListModelsHandlerForTest(catalog *modelcatalog.ModelCatalog, store configstore.ConfigStore, manager *mockModelsManager, providers ...schemas.ModelProvider) *CompletionHandler {
	providerConfigs := make(map[schemas.ModelProvider]configstore.ProviderConfig, len(providers))
	for _, provider := range providers {
		providerConfigs[provider] = configstore.ProviderConfig{}
	}
	return &CompletionHandler{
		modelsManager: manager,
		config: &lib.Config{
			Providers:    providerConfigs,
			ModelCatalog: catalog,
			ConfigStore:  store,
		},
	}
}

func newRoutingConfigStoreForTest(t *testing.T, rules ...configstoreTables.TableRoutingRule) configstore.ConfigStore {
	t.Helper()

	store, err := configstore.NewConfigStore(context.Background(), &configstore.Config{
		Enabled: true,
		Type:    configstore.ConfigStoreTypeSQLite,
		Config:  &configstore.SQLiteConfig{Path: filepath.Join(t.TempDir(), "list_models_routing.db")},
	}, &mockLogger{})
	if err != nil {
		t.Fatalf("failed to create config store: %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close(context.Background())
	})

	for i := range rules {
		rule := rules[i]
		if rule.Name == "" {
			rule.Name = rule.ID
		}
		if rule.Scope == "" {
			rule.Scope = "global"
		}
		if rule.Enabled == nil {
			rule.Enabled = bifrost.Ptr(true)
		}
		if err := store.CreateRoutingRule(context.Background(), &rule); err != nil {
			t.Fatalf("failed to create routing rule %q: %v", rule.ID, err)
		}
	}

	return store
}

func routingRule(id, expr string, priority int, scope string, scopeID *string, targets ...configstoreTables.TableRoutingTarget) configstoreTables.TableRoutingRule {
	return configstoreTables.TableRoutingRule{
		ID:            id,
		Name:          id,
		Enabled:       bifrost.Ptr(true),
		CelExpression: expr,
		Scope:         scope,
		ScopeID:       scopeID,
		Priority:      priority,
		Targets:       targets,
	}
}

func routingTarget(provider, model string) configstoreTables.TableRoutingTarget {
	var providerPtr, modelPtr *string
	if provider != "" {
		providerPtr = bifrost.Ptr(provider)
	}
	if model != "" {
		modelPtr = bifrost.Ptr(model)
	}
	return configstoreTables.TableRoutingTarget{
		Provider: providerPtr,
		Model:    modelPtr,
		Weight:   1.0,
	}
}

func modelIDs(models []schemas.Model) []string {
	ids := make([]string, len(models))
	for i, model := range models {
		ids[i] = model.ID
	}
	return ids
}

func modelByID(models []schemas.Model, id string) (schemas.Model, bool) {
	for _, model := range models {
		if model.ID == id {
			return model, true
		}
	}
	return schemas.Model{}, false
}

func TestExtractListModelsRoutingRuleModels_SupportsStaticModelClauses(t *testing.T) {
	got := extractListModelsRoutingRuleModels("model == 'gpt-5.4' || model in ['best-model', 'gpt-5.4', 'best-model']")
	want := []string{"gpt-5.4", "best-model"}
	if !slices.Equal(got, want) {
		t.Fatalf("routing models = %#v, want %#v", got, want)
	}
}

func TestExtractListModelsRoutingRuleModels_RejectsDynamicOrProviderScopedClauses(t *testing.T) {
	tests := []string{
		"request_type == 'chat_completion' && model == 'gpt-5.4'",
		"model == 'openai/gpt-4o'",
		"model in ['gpt-5.4', 7]",
	}

	for _, expr := range tests {
		t.Run(expr, func(t *testing.T) {
			if got := extractListModelsRoutingRuleModels(expr); got != nil {
				t.Fatalf("routing models = %#v, want nil", got)
			}
		})
	}
}

func TestMergeRoutingListModelsResponse_AddsBareRoutingModelsAndPreservesLiveRows(t *testing.T) {
	catalog := modelCatalogForPricingJSON(t, []byte(`{
		"gpt-4o": {
			"provider": "openai",
			"mode": "chat",
			"input_cost_per_token": 0.0000025,
			"output_cost_per_token": 0.00001
		}
	}`))
	store := newRoutingConfigStoreForTest(t,
		routingRule("rule-best-model", "model == 'best-model'", 0, "global", nil, routingTarget("openai", "gpt-4o")),
	)
	manager := &mockModelsManager{}
	h := newListModelsHandlerForTest(catalog, store, manager, schemas.OpenAI)

	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{{
		ID:               "openai/gpt-4o",
		SupportedMethods: []string{"chat.completions"},
	}}}

	resp = h.mergeRoutingListModelsResponse(nil, "", resp)
	enrichListModelsResponse(resp, catalog)
	sort.Slice(resp.Data, func(i, j int) bool { return resp.Data[i].ID < resp.Data[j].ID })

	if got, want := modelIDs(resp.Data), []string{"best-model", "openai/gpt-4o"}; !slices.Equal(got, want) {
		t.Fatalf("merged ids = %#v, want %#v", got, want)
	}

	liveModel, ok := modelByID(resp.Data, "openai/gpt-4o")
	if !ok {
		t.Fatalf("expected live gpt-4o row")
	}
	if got, want := liveModel.SupportedMethods, []string{"chat.completions"}; !slices.Equal(got, want) {
		t.Fatalf("supported_methods = %#v, want %#v", got, want)
	}

	routingModel, ok := modelByID(resp.Data, "best-model")
	if !ok {
		t.Fatalf("expected synthetic routing row")
	}
	if routingModel.Alias == nil || *routingModel.Alias != "openai/gpt-4o" {
		t.Fatalf("routing alias = %#v, want openai/gpt-4o", routingModel.Alias)
	}
	if routingModel.Pricing == nil || routingModel.Pricing.Prompt == nil || *routingModel.Pricing.Prompt != "0.0000025000" {
		t.Fatalf("routing model prompt pricing = %#v, want 0.0000025000", routingModel.Pricing)
	}
}

func TestMergeRoutingListModelsResponse_UsesHighestPrecedenceRulePerModelAndRespectsAccess(t *testing.T) {
	teamID := "team-1"
	store := newRoutingConfigStoreForTest(t,
		routingRule("team-best-model", "model == 'best-model'", 0, "team", &teamID, routingTarget("anthropic", "claude-3-7-sonnet")),
		routingRule("global-best-model", "model == 'best-model'", 0, "global", nil, routingTarget("openai", "gpt-4o")),
		routingRule("global-only", "model == 'global-only'", 1, "global", nil, routingTarget("openai", "gpt-4o")),
	)

	permit := grant.NewPermit(grant.PermitVirtualKey, "vk-test", "Test VK", true, false, []schemas.ProviderPermit{{
		Provider:      string(schemas.OpenAI),
		AllowedModels: []string{"gpt-4o"},
	}}, nil)
	manager := &mockModelsManager{access: grant.NewAccess([]schemas.Permit{permit}, nil, "", nil)}
	h := newListModelsHandlerForTest(nil, store, manager, schemas.OpenAI, schemas.Anthropic)

	bifrostCtx := schemas.NewBifrostContext(context.Background(), time.Time{})
	bifrostCtx.SetValue(schemas.BifrostContextKeyGovernanceTeamID, teamID)

	resp := h.mergeRoutingListModelsResponse(bifrostCtx, "", &schemas.BifrostListModelsResponse{})
	if got, want := modelIDs(resp.Data), []string{"global-only"}; !slices.Equal(got, want) {
		t.Fatalf("merged ids = %#v, want %#v", got, want)
	}
	if manager.resolveCalls != 1 {
		t.Fatalf("resolve access calls = %d, want 1", manager.resolveCalls)
	}
}

func TestMergeRoutingListModelsResponse_DoesNotAugmentProviderScopedRequests(t *testing.T) {
	store := newRoutingConfigStoreForTest(t,
		routingRule("rule-best-model", "model == 'best-model'", 0, "global", nil, routingTarget("openai", "gpt-4o")),
	)
	h := newListModelsHandlerForTest(nil, store, &mockModelsManager{}, schemas.OpenAI)

	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{{ID: "openai/gpt-4o"}}}
	resp = h.mergeRoutingListModelsResponse(nil, schemas.OpenAI, resp)

	if got, want := modelIDs(resp.Data), []string{"openai/gpt-4o"}; !slices.Equal(got, want) {
		t.Fatalf("merged ids = %#v, want %#v", got, want)
	}
}

func TestCloneAndMergeRoutingListModelsResponse_PreservesErrorContextForRoutingFallback(t *testing.T) {
	liveErr := &schemas.BifrostError{
		Error: &schemas.ErrorField{Message: "provider list failed"},
		ExtraFields: schemas.BifrostErrorExtraFields{
			RequestType: schemas.ListModelsRequest,
			Provider:    schemas.OpenAI,
			Latency:     42,
			KeyStatuses: []schemas.KeyStatus{{
				Provider: schemas.OpenAI,
				KeyID:    "key-a",
				Status:   schemas.KeyStatusListModelsFailed,
			}},
		},
	}
	store := newRoutingConfigStoreForTest(t,
		routingRule("rule-best-model", "model == 'best-model'", 0, "global", nil, routingTarget("openai", "gpt-4o")),
	)
	h := newListModelsHandlerForTest(nil, store, &mockModelsManager{}, schemas.OpenAI)

	resp := cloneListModelsResponse(nil, liveErr)
	resp = h.mergeRoutingListModelsResponse(nil, "", resp)

	if got, want := modelIDs(resp.Data), []string{"best-model"}; !slices.Equal(got, want) {
		t.Fatalf("merged ids = %#v, want %#v", got, want)
	}
	if got, want := resp.ExtraFields.RequestType, schemas.ListModelsRequest; got != want {
		t.Fatalf("request type = %q, want %q", got, want)
	}
	if got, want := resp.ExtraFields.Provider, schemas.OpenAI; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := resp.ExtraFields.Latency, int64(42); got != want {
		t.Fatalf("latency = %d, want %d", got, want)
	}
	if len(resp.KeyStatuses) != 1 {
		t.Fatalf("key statuses = %#v, want one entry", resp.KeyStatuses)
	}
	if got, want := resp.KeyStatuses[0].Status, schemas.KeyStatusListModelsFailed; got != want {
		t.Fatalf("key status = %q, want %q", got, want)
	}
}
