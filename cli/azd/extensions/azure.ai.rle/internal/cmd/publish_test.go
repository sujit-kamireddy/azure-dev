// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"azure.ai.rle/internal/project"

	"github.com/azure/azure-dev/cli/azd/pkg/azdext"
)

func TestMain(m *testing.M) {
	// The publish regression test uses this binary instead of a real container runtime.
	if os.Getenv("RLE_TEST_PUBLISH_CONTAINER_RUNTIME") == "1" {
		if len(os.Args) > 1 && (os.Args[1] == "build" || os.Args[1] == "push") {
			os.Exit(0)
		}
		os.Exit(1)
	}
	os.Exit(m.Run())
}

func TestPublishDisplaysReleaseIdentityAndPreservesJSON(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	t.Setenv("AZURE_CONTAINER_REGISTRY_ENDPOINT", "example.azurecr.io")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("AZD_CONTAINER_RUNTIME", executable)
	t.Setenv("RLE_TEST_PUBLISH_CONTAINER_RUNTIME", "1")
	config := project.RleConfig{
		SchemaVersion: new(project.CurrentRleManifestSchemaVersion),
		Rle: project.RleManifest{
			Name: "math_rl", Version: "1.0.0",
			Type: project.RleTypeGym, Subtype: project.RleSubtypeOpenEnv,
		},
		Defaults: &project.RleEnvironmentDefaults{
			Model: &project.RleModelDefaults{Name: new("test-model")},
		},
	}
	if err := project.WriteRleConfig(dir, config); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "Dockerfile"), []byte("FROM scratch\n"), 0600); err != nil {
		t.Fatal(err)
	}
	published := environmentResource{
		Id: "opaque-environment-id", Name: config.Rle.Name, Version: config.Rle.Version,
		Type: "Gym", Subtype: "OpenEnv",
		AcrImagePath: "example.azurecr.io/project-1-math-rl:1.0.0",
		Defaults:     config.Defaults, CreatedAt: "2026-07-30T04:00:00Z", UpdatedAt: "2026-07-30T05:00:00Z",
		SchemaVersion: config.SchemaVersion,
	}
	requestCount := 0
	controlPlane := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		switch {
		case r.Method == http.MethodGet && r.URL.Path == testFoundryProjectPath+environmentCollectionPath+"/math_rl":
			http.NotFound(w, r)
		case r.Method == http.MethodPost && r.URL.Path == testFoundryProjectPath+environmentCollectionPath:
			var request v1EnvironmentRequest
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Error(err)
				return
			}
			if request.Name != published.Name || request.Version != published.Version ||
				request.Type != published.Type || request.Subtype != published.Subtype ||
				request.AcrImagePath != published.AcrImagePath {
				t.Errorf("unexpected publish request: %#v", request)
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(published); err != nil {
				t.Error(err)
			}
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer controlPlane.Close()
	stubRleClientEndpoint(t, controlPlane.URL)

	command := newPublishCommand()
	var stdout, stderr bytes.Buffer
	command.SetOut(&stdout)
	command.SetErr(&stderr)
	if err := command.Execute(); err != nil {
		t.Fatalf("publish failed: %v; stderr: %s", err, stderr.String())
	}
	if requestCount != 2 {
		t.Fatalf("expected preflight and publish requests, got %d", requestCount)
	}
	const success = "\nPublished environment 'math_rl' version 1.0.0\n"
	_, details, found := strings.Cut(stdout.String(), success)
	if !found {
		t.Fatalf("expected exact success line %q, got %s", success, stdout.String())
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(details), &payload); err != nil {
		t.Fatalf("expected JSON after success line, got %s: %v", details, err)
	}
	for field, want := range map[string]string{
		"environmentId": published.Id, "environmentName": published.Name, "environmentVersion": published.Version,
		"type": published.Type, "subtype": published.Subtype, "acrImage": published.AcrImagePath,
		"foundryProjectEndpoint": "https://account.services.ai.azure.com/api/projects/project-1",
		"createdAt":              published.CreatedAt, "updatedAt": published.UpdatedAt,
		"schemaVersion": *published.SchemaVersion,
	} {
		if payload[field] != want {
			t.Errorf("%s = %v, want %q", field, payload[field], want)
		}
	}
	if defaults, ok := payload["defaults"].(map[string]any); !ok || defaults["model"] == nil {
		t.Fatalf("expected defaults to remain in JSON, got %#v", payload["defaults"])
	}
	t.Logf("Captured publish output:\n%s", stdout.String())
}

func TestBuildEnvironmentCreateRequestMapsManifestConfiguration(t *testing.T) {
	agentName := "support-agent"
	agentVersion := "12"
	schemaVersion := project.CurrentRleManifestSchemaVersion
	modelName := "Qwen/Qwen3-32B"
	numberOfEpochs := 3
	config := project.RleConfig{
		SchemaVersion: &schemaVersion,
		Rle: project.RleManifest{
			Name:         "support_rle",
			Version:      "1.0.1",
			Type:         project.RleTypeHarness,
			Subtype:      project.RleSubtypeHostedAgent,
			AgentName:    &agentName,
			AgentVersion: &agentVersion,
		},
		Defaults: &project.RleEnvironmentDefaults{
			Model: &project.RleModelDefaults{Name: &modelName},
			Reinforcement: &project.RleReinforcementDefaults{
				Hyperparameters: &project.RleReinforcementHyperparameters{
					NumberOfEpochs: &numberOfEpochs,
				},
			},
		},
	}

	request := buildEnvironmentCreateRequest(config, "example.azurecr.io/support_rle:1.0.1")
	if request.Name != "support_rle" ||
		request.Version != "1.0.1" ||
		request.Type != "Harness" ||
		request.Subtype != "HostedAgent" ||
		request.AgentName == nil || *request.AgentName != agentName ||
		request.AgentVersion == nil || *request.AgentVersion != agentVersion ||
		request.SchemaVersion == nil || *request.SchemaVersion != project.CurrentRleManifestSchemaVersion ||
		request.Defaults == nil || request.Defaults.Model == nil ||
		request.Defaults.Model.Name == nil || *request.Defaults.Model.Name != modelName ||
		request.BaseURL != nil {
		t.Fatalf("expected manifest data to map to create request, got %#v", request)
	}

	data, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatal(err)
	}
	for key, expected := range map[string]string{
		"name":          "support_rle",
		"version":       "1.0.1",
		"schemaVersion": project.CurrentRleManifestSchemaVersion,
		"type":          "Harness",
		"subtype":       "HostedAgent",
		"agentName":     agentName,
		"agentVersion":  agentVersion,
	} {
		if payload[key] != expected {
			t.Fatalf("expected %s=%q, got %#v", key, expected, payload[key])
		}
	}
	if _, exists := payload["baseUrl"]; exists {
		t.Fatalf("expected HostedAgent request to omit baseUrl, got %s", data)
	}
	if _, exists := payload["versionBump"]; exists {
		t.Fatalf("expected explicit version request to omit versionBump, got %s", data)
	}
	if _, exists := payload["metadata"]; exists {
		t.Fatalf("expected request to omit metadata, got %s", data)
	}
	defaults, ok := payload["defaults"].(map[string]any)
	if !ok {
		t.Fatalf("expected defaults payload, got %#v", payload["defaults"])
	}
	model, ok := defaults["model"].(map[string]any)
	if !ok || model["name"] != modelName {
		t.Fatalf("expected model default, got %#v", defaults["model"])
	}
	reinforcement, ok := defaults["reinforcement"].(map[string]any)
	if !ok {
		t.Fatalf("expected reinforcement defaults, got %#v", defaults["reinforcement"])
	}
	hyperparameters, ok := reinforcement["hyperparameters"].(map[string]any)
	if !ok || hyperparameters["n_epochs"] != float64(numberOfEpochs) {
		t.Fatalf("expected snake_case hyperparameter payload, got %#v", reinforcement["hyperparameters"])
	}
}

func TestBuildEnvironmentCreateRequestMapsByohHarnessConfiguration(t *testing.T) {
	baseURL := "https://harness.example.com/rollouts/"
	config := project.RleConfig{
		Rle: project.RleManifest{
			Name:    "customer_harness",
			Version: "1.0.1",
			Type:    project.RleTypeHarness,
			Subtype: project.RleSubtypeBYOH,
			BaseURL: &baseURL,
		},
	}

	request := buildEnvironmentCreateRequest(config, "example.azurecr.io/customer_harness:1.0.1")
	if request.Type != "Harness" ||
		request.Subtype != "BYOH" ||
		request.BaseURL == nil || *request.BaseURL != baseURL ||
		request.AgentName != nil ||
		request.AgentVersion != nil {
		t.Fatalf("expected BYOH manifest data to map to create request, got %#v", request)
	}
}

func TestResolvePublishImageUsesManifestVersion(t *testing.T) {
	t.Setenv("AZURE_CONTAINER_REGISTRY_ENDPOINT", "example.azurecr.io")

	image, err := resolvePublishImage(
		"code_rl",
		"1.2.0",
		"https://account.services.ai.azure.com/api/projects/project-name",
	)
	if err != nil {
		t.Fatal(err)
	}
	if image != "example.azurecr.io/project-name-code-rl:1.2.0" {
		t.Fatalf("unexpected versioned image name %q", image)
	}
}

func TestResolvePublishImageRequiresAcrRegistryForManifest(t *testing.T) {
	t.Setenv("AZURE_CONTAINER_REGISTRY_ENDPOINT", "")
	_, err := resolvePublishImage(
		"code_rl",
		"1.0.0",
		"https://account.services.ai.azure.com/api/projects/project-name",
	)
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_acr_registry_required" {
		t.Fatalf("expected ACR registry error, got %v", err)
	}
}

func TestVerifyPublishedEnvironmentRequiresManifestIdentity(t *testing.T) {
	config := project.RleConfig{Rle: project.RleManifest{
		Name:    "code_rl",
		Version: "1.0.0",
		Type:    project.RleTypeGym,
		Subtype: project.RleSubtypeOpenEnv,
	}}
	err := verifyPublishedEnvironment(config, &environmentResource{
		Name:    "code_rl",
		Version: "1.0.1",
		Type:    "Gym",
		Subtype: "OpenEnv",
	})
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_published_environment_mismatch" {
		t.Fatalf("expected published identity mismatch, got %v", err)
	}
}

func TestVerifyPublishedEnvironmentRequiresManifestDefaults(t *testing.T) {
	schemaVersion := project.CurrentRleManifestSchemaVersion
	modelName := "Qwen/Qwen3-32B"
	differentModelName := "Qwen/Qwen3-14B"
	config := project.RleConfig{
		SchemaVersion: &schemaVersion,
		Rle: project.RleManifest{
			Name:    "code_rl",
			Version: "1.0.0",
			Type:    project.RleTypeGym,
			Subtype: project.RleSubtypeOpenEnv,
		},
		Defaults: &project.RleEnvironmentDefaults{
			Model: &project.RleModelDefaults{Name: &modelName},
		},
	}
	err := verifyPublishedEnvironment(config, &environmentResource{
		Name:          "code_rl",
		Version:       "1.0.0",
		Type:          "Gym",
		Subtype:       "OpenEnv",
		SchemaVersion: &schemaVersion,
		Defaults: &project.RleEnvironmentDefaults{
			Model: &project.RleModelDefaults{Name: &differentModelName},
		},
	})
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_published_environment_mismatch" {
		t.Fatalf("expected published defaults mismatch, got %v", err)
	}
}

func TestPublishRequestSendsTheEnvironmentProtocolUnderItsWireName(t *testing.T) {
	config := project.RleConfig{Rle: project.RleManifest{
		Name:                "competitive_intelligence_agent",
		Version:             "2.0.0",
		Type:                project.RleTypeHarness,
		Subtype:             project.RleSubtypeHostedAgent,
		AgentName:           new("ci-agent"),
		AgentVersion:        new("15"),
		EnvironmentProtocol: new(project.RleEnvironmentProtocolMcpEnvironment),
	}}

	body, err := json.Marshal(buildEnvironmentCreateRequest(config, "registry.azurecr.io/ci:2.0.0"))
	if err != nil {
		t.Fatal(err)
	}
	var sent map[string]any
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatal(err)
	}
	// The service deserializes this one key as snake_case while its neighbours
	// are camelCase, so the name is asserted literally rather than through the
	// struct: a camelCase slip would be dropped and publish would succeed with
	// a legacy environment.
	if sent["environment_protocol"] != "mcp_environment" {
		t.Fatalf("environment_protocol = %v, want mcp_environment; body %s", sent["environment_protocol"], body)
	}

	config.Rle.EnvironmentProtocol = nil
	body, err = json.Marshal(buildEnvironmentCreateRequest(config, "registry.azurecr.io/ci:2.0.0"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "environment_protocol") {
		t.Fatalf("a legacy environment must omit the key entirely, got %s", body)
	}
}

func TestVerifyPublishedEnvironmentRejectsADowngradedProtocol(t *testing.T) {
	config := project.RleConfig{Rle: project.RleManifest{
		Name:                "competitive_intelligence_agent",
		Version:             "2.0.0",
		Type:                project.RleTypeHarness,
		Subtype:             project.RleSubtypeHostedAgent,
		AgentName:           new("ci-agent"),
		AgentVersion:        new("15"),
		EnvironmentProtocol: new(project.RleEnvironmentProtocolMcpEnvironment),
	}}
	published := &environmentResource{
		Name:         "competitive_intelligence_agent",
		Version:      "2.0.0",
		Type:         "Harness",
		Subtype:      "HostedAgent",
		AgentName:    "ci-agent",
		AgentVersion: "15",
	}

	// A version's protocol is immutable, so a service that ignored the field
	// leaves the user with an environment that can never take the MCP path.
	// Without this check publish would report success.
	err := verifyPublishedEnvironment(config, published)
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_published_environment_mismatch" {
		t.Fatalf("expected a protocol mismatch, got %v", err)
	}
	if !strings.Contains(localErr.Message, "legacy (unset)") {
		t.Fatalf("message should name the absent protocol, got %q", localErr.Message)
	}

	published.EnvironmentProtocol = new(project.RleEnvironmentProtocolMcpEnvironment)
	if err := verifyPublishedEnvironment(config, published); err != nil {
		t.Fatalf("a matching protocol must verify, got %v", err)
	}
}

func TestPublishTopLevelMcpGymManifest(t *testing.T) {
	dir := t.TempDir()
	content := `environment_protocol = "mcp_environment"

[rle]
name = "math_rl"
version = "1.0.0"
type = "Gym"
subtype = "OpenEnv"
`
	if err := os.WriteFile(filepath.Join(dir, project.RleConfigFile), []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	config, err := project.LoadRleConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(buildEnvironmentCreateRequest(config, "registry.azurecr.io/math:1.0.0"))
	if err != nil {
		t.Fatal(err)
	}
	var sent map[string]any
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatal(err)
	}
	if sent["environment_protocol"] != "mcp_environment" || sent["type"] != "Gym" || sent["subtype"] != "OpenEnv" {
		t.Fatalf("MCP Gym manifest did not reach the publish request: %s", body)
	}
	published := &environmentResource{
		Name: "math_rl", Version: "1.0.0", Type: "Gym", Subtype: "OpenEnv",
		EnvironmentProtocol: new(project.RleEnvironmentProtocolMcpEnvironment),
	}
	if err := verifyPublishedEnvironment(config, published); err != nil {
		t.Fatal(err)
	}
	published.EnvironmentProtocol = nil
	if err := verifyPublishedEnvironment(config, published); err == nil {
		t.Fatal("publishing MCP Gym as legacy must not report success")
	}
}

func TestPublishRequiresManifestBeforeProjectConfiguration(t *testing.T) {
	tempDir := t.TempDir()
	t.Chdir(tempDir)
	t.Setenv(foundryProjectEndpointEnvVar, "")

	command := newPublishCommand()
	err := command.Execute()
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_manifest_missing" {
		t.Fatalf("expected missing manifest error, got %v", err)
	}
}

func TestEnvironmentOutputUsesEnvironmentNameField(t *testing.T) {
	body, err := json.Marshal(environmentOutput{
		EnvironmentId:      "env-1",
		EnvironmentVersion: "1.0.0",
		EnvironmentName:    "echo_env",
		Type:               "Gym",
		Subtype:            "OpenEnv",
	})
	if err != nil {
		t.Fatal(err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["environmentName"] != "echo_env" {
		t.Fatalf("expected environmentName field, got %v", payload)
	}
	if _, exists := payload["name"]; exists {
		t.Fatalf("expected legacy name field to be omitted, got %v", payload)
	}
}

func TestResolvePublishTargetRequiresInitialManifestVersion(t *testing.T) {
	tempDir := t.TempDir()
	t.Chdir(tempDir)
	t.Setenv(foundryProjectEndpointEnvVar, "https://account.services.ai.azure.com/api/projects/project")
	if err := project.WriteRleConfig(tempDir, project.RleConfig{
		Rle: project.RleManifest{
			Name:    "code_rl",
			Version: "0.1.0",
			Type:    project.RleTypeGym,
			Subtype: project.RleSubtypeOpenEnv,
		},
	}); err != nil {
		t.Fatal(err)
	}

	controlPlane := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet ||
			r.URL.Path != testFoundryProjectPath+environmentCollectionPath+"/code_rl" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		http.NotFound(w, r)
	}))
	defer controlPlane.Close()
	stubRleClientEndpoint(t, controlPlane.URL)

	_, _, _, _, err := resolvePublishTarget(t.Context())
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_manifest_initial_version_invalid" {
		t.Fatalf("expected invalid initial version error, got %v", err)
	}

	if _, err := project.LoadRleConfig(tempDir); err != nil {
		t.Fatalf("expected manifest to remain unchanged after failed preflight: %v", err)
	}
}
