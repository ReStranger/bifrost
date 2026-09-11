package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

func TestApplyCatalogCapabilityMetadata_DerivesSurfaceFromModeWithoutClobberingProviderMethods(t *testing.T) {
	modality := "text"
	model := schemas.Model{
		SupportedMethods: []string{"provider-native"},
	}
	capability := &modelcatalog.PricingEntry{
		Mode:            "responses",
		ContextLength:   intPtr(200000),
		MaxInputTokens:  intPtr(128000),
		MaxOutputTokens: intPtr(32000),
		Architecture: &schemas.Architecture{
			Modality: &modality,
		},
	}

	applyCatalogCapabilityMetadata(&model, capability)

	if model.Mode == nil || *model.Mode != "responses" {
		t.Fatalf("expected mode=responses, got %#v", model.Mode)
	}
	if len(model.SupportedEndpoints) != 1 || model.SupportedEndpoints[0] != "/v1/responses" {
		t.Fatalf("expected supported_endpoints=[/v1/responses], got %#v", model.SupportedEndpoints)
	}
	if len(model.SupportedMethods) != 1 || model.SupportedMethods[0] != "provider-native" {
		t.Fatalf("expected provider-native supported_methods to be preserved, got %#v", model.SupportedMethods)
	}
	if model.ContextLength == nil || *model.ContextLength != 200000 {
		t.Fatalf("expected context_length=200000, got %#v", model.ContextLength)
	}
	if model.MaxInputTokens == nil || *model.MaxInputTokens != 128000 {
		t.Fatalf("expected max_input_tokens=128000, got %#v", model.MaxInputTokens)
	}
	if model.MaxOutputTokens == nil || *model.MaxOutputTokens != 32000 {
		t.Fatalf("expected max_output_tokens=32000, got %#v", model.MaxOutputTokens)
	}
	if model.Architecture == nil || model.Architecture.Modality == nil || *model.Architecture.Modality != modality {
		t.Fatalf("expected architecture modality=%q, got %#v", modality, model.Architecture)
	}
}

func TestApplyCatalogCapabilityMetadata_BackfillsBifrostRequestTypesWhenProviderMethodsMissing(t *testing.T) {
	testCases := []struct {
		name     string
		mode     string
		endpoint string
		method   string
	}{
		{
			name:     "responses",
			mode:     "responses",
			endpoint: "/v1/responses",
			method:   string(schemas.ResponsesRequest),
		},
		{
			name:     "chat",
			mode:     "chat",
			endpoint: "/v1/chat/completions",
			method:   string(schemas.ChatCompletionRequest),
		},
		{
			name:     "audio speech",
			mode:     "audio_speech",
			endpoint: "/v1/audio/speech",
			method:   string(schemas.SpeechRequest),
		},
		{
			name:     "image generation",
			mode:     "image_generation",
			endpoint: "/v1/images/generations",
			method:   string(schemas.ImageGenerationRequest),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model := schemas.Model{}
			capability := &modelcatalog.PricingEntry{Mode: tc.mode}

			applyCatalogCapabilityMetadata(&model, capability)

			if model.Mode == nil || *model.Mode != tc.mode {
				t.Fatalf("expected mode=%s, got %#v", tc.mode, model.Mode)
			}
			if len(model.SupportedEndpoints) != 1 || model.SupportedEndpoints[0] != tc.endpoint {
				t.Fatalf("expected supported_endpoints=[%s], got %#v", tc.endpoint, model.SupportedEndpoints)
			}
			if len(model.SupportedMethods) != 1 || model.SupportedMethods[0] != tc.method {
				t.Fatalf("expected supported_methods=[%s], got %#v", tc.method, model.SupportedMethods)
			}
		})
	}
}

func TestApplyCatalogCapabilityMetadata_SkipsUnknownModeSurfaceBackfill(t *testing.T) {
	model := schemas.Model{}
	capability := &modelcatalog.PricingEntry{
		Mode:          "unknown",
		ContextLength: intPtr(4096),
	}

	applyCatalogCapabilityMetadata(&model, capability)

	if model.ContextLength == nil || *model.ContextLength != 4096 {
		t.Fatalf("expected context_length=4096, got %#v", model.ContextLength)
	}
	if model.Mode != nil {
		t.Fatalf("expected mode to remain nil for unknown mode, got %#v", model.Mode)
	}
	if len(model.SupportedEndpoints) != 0 {
		t.Fatalf("expected supported_endpoints to remain empty, got %#v", model.SupportedEndpoints)
	}
	if len(model.SupportedMethods) != 0 {
		t.Fatalf("expected supported_methods to remain empty, got %#v", model.SupportedMethods)
	}
}

func TestApplyCatalogCapabilityMetadata_PreservesExistingSurface(t *testing.T) {
	existingMode := "responses"
	model := schemas.Model{
		Mode:               &existingMode,
		SupportedEndpoints: []string{"/v1/responses"},
		SupportedMethods:   []string{"provider-native"},
	}
	capability := &modelcatalog.PricingEntry{
		Mode:          "chat",
		ContextLength: intPtr(8192),
	}

	applyCatalogCapabilityMetadata(&model, capability)

	if model.Mode == nil || *model.Mode != existingMode {
		t.Fatalf("expected existing mode=%s to be preserved, got %#v", existingMode, model.Mode)
	}
	if len(model.SupportedEndpoints) != 1 || model.SupportedEndpoints[0] != "/v1/responses" {
		t.Fatalf("expected existing supported_endpoints to be preserved, got %#v", model.SupportedEndpoints)
	}
	if len(model.SupportedMethods) != 1 || model.SupportedMethods[0] != "provider-native" {
		t.Fatalf("expected existing supported_methods to be preserved, got %#v", model.SupportedMethods)
	}
	if model.ContextLength == nil || *model.ContextLength != 8192 {
		t.Fatalf("expected context_length=8192, got %#v", model.ContextLength)
	}
}

func TestApplyCatalogCapabilityMetadata_PreservesProviderMethodsWithoutCatalogEntry(t *testing.T) {
	model := schemas.Model{
		SupportedMethods: []string{"provider-native"},
	}

	applyCatalogCapabilityMetadata(&model, nil)

	if model.Mode != nil {
		t.Fatalf("expected mode to remain nil, got %#v", model.Mode)
	}
	if len(model.SupportedEndpoints) != 0 {
		t.Fatalf("expected supported_endpoints to remain empty, got %#v", model.SupportedEndpoints)
	}
	if len(model.SupportedMethods) != 1 || model.SupportedMethods[0] != "provider-native" {
		t.Fatalf("expected provider-native supported_methods to be preserved, got %#v", model.SupportedMethods)
	}
}

func TestEnrichListModelsResponse_BackfillsCapabilitiesFromAliasEntry(t *testing.T) {
	modality := "text"
	catalog := newHTTPBackedTestModelCatalog(t, map[string]modelcatalog.PricingEntry{
		"deployment-model": {
			Provider:        string(schemas.OpenAI),
			Mode:            "chat",
			ContextLength:   intPtr(16384),
			MaxInputTokens:  intPtr(12000),
			MaxOutputTokens: intPtr(4096),
			Architecture: &schemas.Architecture{
				Modality: &modality,
			},
		},
	})
	resp := &schemas.BifrostListModelsResponse{
		Data: []schemas.Model{{
			ID:      "openai/alias-model",
			Alias:   schemas.Ptr("deployment-model"),
			Pricing: &schemas.Pricing{Prompt: schemas.Ptr("existing")},
		}},
	}

	enrichListModelsResponse(resp, catalog)

	model := resp.Data[0]
	if model.Mode == nil || *model.Mode != "chat" {
		t.Fatalf("expected mode=chat from alias fallback, got %#v", model.Mode)
	}
	if len(model.SupportedEndpoints) != 1 || model.SupportedEndpoints[0] != "/v1/chat/completions" {
		t.Fatalf("expected supported_endpoints=[/v1/chat/completions], got %#v", model.SupportedEndpoints)
	}
	if len(model.SupportedMethods) != 1 || model.SupportedMethods[0] != string(schemas.ChatCompletionRequest) {
		t.Fatalf("expected supported_methods=[%s], got %#v", string(schemas.ChatCompletionRequest), model.SupportedMethods)
	}
	if model.ContextLength == nil || *model.ContextLength != 16384 {
		t.Fatalf("expected context_length=16384, got %#v", model.ContextLength)
	}
	if model.MaxInputTokens == nil || *model.MaxInputTokens != 12000 {
		t.Fatalf("expected max_input_tokens=12000, got %#v", model.MaxInputTokens)
	}
	if model.MaxOutputTokens == nil || *model.MaxOutputTokens != 4096 {
		t.Fatalf("expected max_output_tokens=4096, got %#v", model.MaxOutputTokens)
	}
	if model.Architecture == nil || model.Architecture.Modality == nil || *model.Architecture.Modality != modality {
		t.Fatalf("expected architecture modality=%q, got %#v", modality, model.Architecture)
	}
	if model.Pricing == nil || model.Pricing.Prompt == nil || *model.Pricing.Prompt != "existing" {
		t.Fatalf("expected existing pricing to be preserved, got %#v", model.Pricing)
	}
}

func TestApplyCatalogPricingMetadata_FillsMissingPricingOnly(t *testing.T) {
	imageCost := 0.03
	cacheRead := 0.004
	model := schemas.Model{}
	pricing := &modelcatalog.PricingEntry{
		InputCostPerToken:       floatPtr(0.1),
		OutputCostPerToken:      floatPtr(0.2),
		InputCostPerImage:       &imageCost,
		CacheReadInputTokenCost: &cacheRead,
	}

	applyCatalogPricingMetadata(&model, pricing)

	if model.Pricing == nil {
		t.Fatal("expected pricing to be backfilled")
	}
	if model.Pricing.Prompt == nil || *model.Pricing.Prompt != "0.1000000000" {
		t.Fatalf("expected prompt price 0.1000000000, got %#v", model.Pricing)
	}
	if model.Pricing.Completion == nil || *model.Pricing.Completion != "0.2000000000" {
		t.Fatalf("expected completion price 0.2000000000, got %#v", model.Pricing)
	}
	if model.Pricing.Image == nil || *model.Pricing.Image != "0.0300000000" {
		t.Fatalf("expected image price 0.0300000000, got %#v", model.Pricing)
	}
	if model.Pricing.InputCacheRead == nil || *model.Pricing.InputCacheRead != "0.0040000000" {
		t.Fatalf("expected input_cache_read price 0.0040000000, got %#v", model.Pricing)
	}

	existingPrompt := "already-set"
	model.Pricing = &schemas.Pricing{Prompt: &existingPrompt}
	applyCatalogPricingMetadata(&model, &modelcatalog.PricingEntry{InputCostPerToken: floatPtr(9), OutputCostPerToken: floatPtr(9)})
	if model.Pricing.Prompt == nil || *model.Pricing.Prompt != existingPrompt {
		t.Fatalf("expected existing pricing to be preserved, got %#v", model.Pricing)
	}
	if model.Pricing.Completion != nil {
		t.Fatalf("expected existing pricing object to remain untouched, got %#v", model.Pricing)
	}
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
	if routingModel.Mode == nil || *routingModel.Mode != "chat" {
		t.Fatalf("routing mode = %#v, want chat", routingModel.Mode)
	}
	if got, want := routingModel.SupportedEndpoints, []string{"/v1/chat/completions"}; !slices.Equal(got, want) {
		t.Fatalf("routing supported_endpoints = %#v, want %#v", got, want)
	}
	if got, want := routingModel.SupportedMethods, []string{string(schemas.ChatCompletionRequest)}; !slices.Equal(got, want) {
		t.Fatalf("routing supported_methods = %#v, want %#v", got, want)
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

func newHTTPBackedTestModelCatalog(t *testing.T, pricingData map[string]modelcatalog.PricingEntry) *modelcatalog.ModelCatalog {
	t.Helper()

	payload, err := json.Marshal(pricingData)
	if err != nil {
		t.Fatalf("failed to marshal pricing data: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(payload)
	}))
	t.Cleanup(server.Close)

	pricingURL := server.URL
	catalog, err := modelcatalog.Init(t.Context(), &modelcatalog.Config{PricingURL: &pricingURL}, nil, &mockLogger{})
	if err != nil {
		t.Fatalf("failed to initialize model catalog: %v", err)
	}
	t.Cleanup(func() {
		if err := catalog.Cleanup(); err != nil {
			t.Errorf("failed to clean up model catalog: %v", err)
		}
	})

	return catalog
}

func intPtr(v int) *int { return &v }

func floatPtr(v float64) *float64 { return &v }
