// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"azure.ai.rle/internal/project"
	"azure.ai.rle/internal/rollouts"
	"github.com/azure/azure-dev/cli/azd/pkg/azdext"
)

func TestBuildFinetuneJobRequestUsesRleEnvironmentMethod(t *testing.T) {
	flags := &rleTrainFlags{
		rleName:    "code_rl",
		rleVersion: "1.0.0",
		model:      "Qwen/Qwen3-32B",
	}

	request := buildFinetuneJobRequest(flags, "file-training", "", nil)

	if request.Model != "Qwen/Qwen3-32B" {
		t.Fatalf("expected model to map from flags, got %q", request.Model)
	}
	if request.TrainingFile != "file-training" {
		t.Fatalf("expected training_file to be set, got %q", request.TrainingFile)
	}
	if request.TrainingType != finetuneTrainingTypeGlobalStandard {
		t.Fatalf("expected GlobalStandard training type, got %d", request.TrainingType)
	}
	if request.Method == nil || request.Method.Type != finetuneMethodTypeRleEnvironment {
		t.Fatalf("expected rl_environment method, got %#v", request.Method)
	}
	if request.Method.RleEnvironment.Name != "code_rl" || request.Method.RleEnvironment.Version != "1.0.0" {
		t.Fatalf("expected rl_environment name/version from flags, got %#v", request.Method.RleEnvironment)
	}
	if request.Method.RleEnvironment.MaxEpisodeSteps != nil {
		t.Fatal("expected max_episode_steps to be omitted when not provided")
	}
	if request.ValidationFile != nil || request.Suffix != nil {
		t.Fatal("expected optional fields to be omitted when not provided")
	}

	data, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"training_file":"file-training"`) {
		t.Fatalf("expected training_file in JSON, got %s", data)
	}
	if !strings.Contains(string(data), `"trainingType":1`) {
		t.Fatalf("expected GlobalStandard trainingType in JSON, got %s", data)
	}
}

func TestBuildFinetuneJobRequestIncludesOptionalFields(t *testing.T) {
	flags := &rleTrainFlags{
		rleName:         "code_rl",
		rleVersion:      "1.0.0",
		model:           "Qwen/Qwen3-32B",
		suffix:          "custom-suffix",
		maxEpisodeSteps: 32,
	}

	request := buildFinetuneJobRequest(flags, "file-abc", "file-def", nil)

	if request.TrainingFile != "file-abc" {
		t.Fatalf("expected training_file to be set, got %q", request.TrainingFile)
	}
	if request.ValidationFile == nil || *request.ValidationFile != "file-def" {
		t.Fatalf("expected validation_file to be set, got %#v", request.ValidationFile)
	}
	if request.Suffix == nil || *request.Suffix != "custom-suffix" {
		t.Fatalf("expected suffix to be set, got %#v", request.Suffix)
	}
	if request.Method.RleEnvironment.MaxEpisodeSteps == nil || *request.Method.RleEnvironment.MaxEpisodeSteps != 32 {
		t.Fatalf("expected max_episode_steps to be set, got %#v", request.Method.RleEnvironment.MaxEpisodeSteps)
	}
}

func TestTrainCommandRequiresTrainingFile(t *testing.T) {
	// rle.toml can supply the training file, so this is no longer a required flag.
	// It is still a required setting, and the error has to name it.
	t.Chdir(t.TempDir())
	cmd := newTrainCommand()
	cmd.SetArgs([]string{
		"--rle-name", "code_rl",
		"--rle-version", "1.0.0",
		"--model", "Qwen/Qwen3-32B",
	})
	cmd.SilenceUsage = true
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	err := cmd.Execute()
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_train_setting_required" {
		t.Fatalf("expected a missing training file error, got %v", err)
	}
	if !strings.Contains(localErr.Message, "training file") {
		t.Fatalf("the error should name the training file: %q", localErr.Message)
	}
}

func TestResolveLocalFilePath(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "training.jsonl")
	if err := os.WriteFile(filePath, []byte("{\"input\":\"example\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name     string
		raw      string
		required bool
		want     string
		wantCode string
	}{
		{name: "trims valid local path", raw: " " + filePath + " ", required: true, want: filePath},
		{name: "allows optional empty path", raw: "  ", required: false, want: ""},
		{name: "rejects missing file", raw: filepath.Join(t.TempDir(), "missing.jsonl"), required: true, wantCode: "rle_train_file_unavailable"},
		{name: "rejects directory", raw: t.TempDir(), required: true, wantCode: "rle_train_file_not_regular"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := resolveLocalFilePath(test.raw, "training", test.required)
			if test.wantCode != "" {
				localErr, ok := errors.AsType[*azdext.LocalError](err)
				if !ok {
					t.Fatalf("expected LocalError, got %T", err)
				}
				if localErr.Code != test.wantCode {
					t.Fatalf("expected error code %q, got %q", test.wantCode, localErr.Code)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("expected %q, got %q", test.want, got)
			}
		})
	}
}

func TestResolveFinetuneEndpointPrefersFlag(t *testing.T) {
	endpoint, err := resolveFinetuneEndpoint(
		"https://from-flag.openai.azure.com",
		"https://account.services.ai.azure.com/api/projects/project",
	)
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "https://from-flag.openai.azure.com" {
		t.Fatalf("expected flag value to take precedence, got %q", endpoint)
	}
}

// The Foundry derivation is not reached while the training service endpoint is
// hardcoded, but it is what the commands go back to once an account's own
// endpoint serves RLE jobs, so it stays covered.
func TestFinetuneEndpointDerivesAccountFromFoundryProject(t *testing.T) {
	endpoint, err := finetuneEndpointFromFoundryProject(
		"https://My-Account.services.ai.azure.com/api/projects/project",
	)
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "https://my-account.openai.azure.com" {
		t.Fatalf("expected endpoint derived from the Foundry account, got %q", endpoint)
	}
}

func TestFinetuneEndpointRejectsInvalidFoundryHost(t *testing.T) {
	_, err := finetuneEndpointFromFoundryProject("https://example.com/api/projects/project")
	localErr, ok := errors.AsType[*azdext.LocalError](err)
	if !ok || localErr.Code != "rle_invalid_project_endpoint" {
		t.Fatalf("expected invalid project endpoint error, got %v", err)
	}
}

// Without a flag the commands must reach the service that actually runs RLE
// jobs, not the Foundry account's own fine-tuning API, which cannot accept them.
func TestResolveFinetuneEndpointDefaultsToTheTrainingService(t *testing.T) {
	t.Setenv(rleTrainEndpointEnvVar, "")

	endpoint, err := resolveFinetuneEndpoint(
		"",
		"https://account.services.ai.azure.com/api/projects/project",
	)
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != rleTrainingServiceEndpoint {
		t.Fatalf("expected the RLE training service, got %q", endpoint)
	}
}

func TestResolveFinetuneEndpointPrefersTheEnvironmentOverTheDefault(t *testing.T) {
	t.Setenv(rleTrainEndpointEnvVar, "https://from-env.example.com")

	endpoint, err := resolveFinetuneEndpoint(
		"",
		"https://account.services.ai.azure.com/api/projects/project",
	)
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "https://from-env.example.com" {
		t.Fatalf("expected %s to take precedence, got %q", rleTrainEndpointEnvVar, endpoint)
	}
}

func TestResolveFinetuneEndpointPrefersTheFlagOverTheEnvironment(t *testing.T) {
	t.Setenv(rleTrainEndpointEnvVar, "https://from-env.example.com")

	endpoint, err := resolveFinetuneEndpoint(
		"https://from-flag.openai.azure.com",
		"https://account.services.ai.azure.com/api/projects/project",
	)
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "https://from-flag.openai.azure.com" {
		t.Fatalf("expected the flag to take precedence, got %q", endpoint)
	}
}

func TestNormalizeFinetuneEndpointRejectsNonHTTPS(t *testing.T) {
	_, err := normalizeFinetuneEndpoint("http://resource.openai.azure.com")
	if err == nil || !strings.Contains(err.Error(), "must use https") {
		t.Fatalf("expected https-only error, got %v", err)
	}
}

func TestFinetuneClientSendsProjectHeadersAndAuthenticates(t *testing.T) {
	credential := &testTokenCredential{}
	client := newFinetuneClientWithCredential("https://resource.openai.azure.com", credential)
	client.httpClient.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if got := request.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Fatalf("expected bearer token, got %q", got)
		}
		if got := request.URL.Path; got != finetuneJobsPath {
			t.Fatalf("expected path %q, got %q", finetuneJobsPath, got)
		}
		if got := request.URL.Query().Get("api-version"); got != "" {
			t.Fatalf("expected no api-version query param, got %q", got)
		}
		if got := request.Header.Get("azureai-project"); got != "myproject" {
			t.Fatalf("expected azureai-project header, got %q", got)
		}
		if got := request.Header.Get("azureai-project-is-default"); got != "true" {
			t.Fatalf("expected azureai-project-is-default=true, got %q", got)
		}
		return &http.Response{
			StatusCode: http.StatusCreated,
			Body:       io.NopCloser(strings.NewReader(`{"id":"ftjob-1","status":"queued"}`)),
			Header:     make(http.Header),
		}, nil
	})

	job, err := client.createJob(t.Context(), finetuneJobCreationRequest{}, "myproject")
	if err != nil {
		t.Fatal(err)
	}
	if job.Id != "ftjob-1" || job.Status != "queued" {
		t.Fatalf("expected decoded job response, got %#v", job)
	}
	if len(credential.scopes) != 1 || credential.scopes[0] != finetuneTokenScope {
		t.Fatalf("expected fine-tuning token scope %q, got %v", finetuneTokenScope, credential.scopes)
	}
}

func TestTrainActionUploadsLocalFileBeforeSubmittingJob(t *testing.T) {
	trainingFilePath := filepath.Join(t.TempDir(), "training.jsonl")
	if err := os.WriteFile(trainingFilePath, []byte("{\"input\":\"example\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(foundryProjectEndpointEnvVar, "https://account.services.ai.azure.com/api/projects/project")

	client := newFinetuneClientWithCredential("https://account.openai.azure.com", &testTokenCredential{})
	uploadCount := 0
	client.httpClient.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch request.URL.Path {
		case finetuneFilesPath:
			uploadCount++
			if _, err := io.ReadAll(request.Body); err != nil {
				t.Fatal(err)
			}
			return &http.Response{
				StatusCode: http.StatusCreated,
				Body:       io.NopCloser(strings.NewReader(`{"id":"file-training"}`)),
				Header:     make(http.Header),
			}, nil
		case finetuneJobsPath:
			if uploadCount != 1 {
				t.Fatalf("expected file upload before job creation, got %d uploads", uploadCount)
			}
			var jobRequest finetuneJobCreationRequest
			if err := json.NewDecoder(request.Body).Decode(&jobRequest); err != nil {
				t.Fatal(err)
			}
			if jobRequest.TrainingFile != "file-training" {
				t.Fatalf("expected uploaded training file ID, got %q", jobRequest.TrainingFile)
			}
			if jobRequest.Method == nil || jobRequest.Method.RleEnvironment.Name != "code_rl" {
				t.Fatalf("expected RLE request, got %#v", jobRequest.Method)
			}
			return &http.Response{
				StatusCode: http.StatusCreated,
				Body:       io.NopCloser(strings.NewReader(`{"id":"ftjob-1","status":"queued"}`)),
				Header:     make(http.Header),
			}, nil
		default:
			t.Fatalf("unexpected request path %q", request.URL.Path)
			return nil, nil
		}
	})

	originalCreateClient := createFinetuneClient
	createFinetuneClient = func(endpoint string) (*finetuneClient, error) {
		if endpoint != rleTrainingServiceEndpoint {
			t.Fatalf("expected the RLE training service, got %q", endpoint)
		}
		return client, nil
	}
	t.Cleanup(func() {
		createFinetuneClient = originalCreateClient
	})

	command := newTrainCommand()
	command.SetContext(context.Background())
	var output bytes.Buffer
	command.SetOut(&output)
	action := &trainAction{
		cmd: command,
		flags: &rleTrainFlags{
			rleName:      "code_rl",
			rleVersion:   "1.0.0",
			model:        "Qwen/Qwen3-32B",
			trainingFile: trainingFilePath,
		},
	}

	if err := action.Run(); err != nil {
		t.Fatal(err)
	}
	if uploadCount != 1 {
		t.Fatalf("expected one uploaded file, got %d", uploadCount)
	}
	if !strings.Contains(output.String(), "Uploaded training file as file-training.") {
		t.Fatalf("expected upload progress output, got %q", output.String())
	}
	if !strings.Contains(output.String(), "Submitted fine-tuning job ftjob-1") {
		t.Fatalf("expected job output, got %q", output.String())
	}
}

func TestFinetuneClientSurfacesHTTPErrors(t *testing.T) {
	client := newFinetuneClientWithCredential("https://resource.openai.azure.com", &testTokenCredential{})
	client.httpClient.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Body:       io.NopCloser(strings.NewReader(`{"error":{"code":"InvalidPayload"}}`)),
			Header:     make(http.Header),
		}, nil
	})

	_, err := client.createJob(t.Context(), finetuneJobCreationRequest{}, "myproject")
	if err == nil {
		t.Fatal("expected an error for HTTP 400")
	}
	wrapped := finetuneServiceError(err)
	serviceErr, ok := errors.AsType[*azdext.ServiceError](wrapped)
	if !ok {
		t.Fatalf("expected a *azdext.ServiceError, got %T", wrapped)
	}
	if serviceErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected status code %d, got %d", http.StatusBadRequest, serviceErr.StatusCode)
	}
	if !strings.Contains(serviceErr.Suggestion, "Loom-eligible") {
		t.Fatalf("expected Loom-eligibility guidance in suggestion, got %v", serviceErr.Suggestion)
	}
}

func writeTrainRleConfig(t *testing.T, dir string, name string, version string) {
	t.Helper()
	schemaVersion := project.CurrentRleManifestSchemaVersion
	if err := project.WriteRleConfig(dir, project.RleConfig{
		SchemaVersion: &schemaVersion,
		Rle: project.RleManifest{
			Name:    name,
			Version: version,
			Type:    project.RleTypeGym,
			Subtype: project.RleSubtypeOpenEnv,
		},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestTrainFallsBackToRleConfigNameAndVersion(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfig(t, dir, "code_rl", "1.2.3")

	action := &trainAction{flags: &rleTrainFlags{model: "m", trainingFile: "t"}}
	if _, err := action.resolveTrainSettings(); err != nil {
		t.Fatal(err)
	}

	if action.flags.rleName != "code_rl" {
		t.Fatalf("expected the name from rle.toml, got %q", action.flags.rleName)
	}
	if action.flags.rleVersion != "1.2.3" {
		t.Fatalf("expected the version from rle.toml, got %q", action.flags.rleVersion)
	}
}

func TestTrainFlagsOverrideRleConfig(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfig(t, dir, "code_rl", "1.2.3")

	action := &trainAction{flags: &rleTrainFlags{rleName: "math_rl", rleVersion: "2.0.0", model: "m", trainingFile: "t"}}
	if _, err := action.resolveTrainSettings(); err != nil {
		t.Fatal(err)
	}

	if action.flags.rleName != "math_rl" || action.flags.rleVersion != "2.0.0" {
		t.Fatalf("expected the flags to win, got %q %q", action.flags.rleName, action.flags.rleVersion)
	}
}

func TestTrainFallsBackToRleConfigVersionOnly(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfig(t, dir, "code_rl", "1.2.3")

	action := &trainAction{flags: &rleTrainFlags{rleName: "math_rl", model: "m", trainingFile: "t"}}
	if _, err := action.resolveTrainSettings(); err != nil {
		t.Fatal(err)
	}

	if action.flags.rleName != "math_rl" || action.flags.rleVersion != "1.2.3" {
		t.Fatalf("expected the version from rle.toml, got %q %q", action.flags.rleName, action.flags.rleVersion)
	}
}

func TestTrainWithoutRleConfigOrFlagsFails(t *testing.T) {
	t.Chdir(t.TempDir())

	action := &trainAction{flags: &rleTrainFlags{}}
	_, err := action.resolveTrainSettings()
	if err == nil {
		t.Fatal("expected an error when neither --rle-name nor rle.toml supplies the environment")
	}

	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_manifest_missing" {
		t.Fatalf("expected a missing rle.toml error, got %v", err)
	}
}

func TestTrainCommandNoLongerRequiresRleFlags(t *testing.T) {
	t.Chdir(t.TempDir())

	cmd := newTrainCommand()
	cmd.SetArgs([]string{"--model", "Qwen/Qwen3-32B"})
	var output bytes.Buffer
	cmd.SetOut(&output)
	cmd.SetErr(&output)

	err := cmd.Execute()
	if err == nil {
		t.Fatal("expected an error when --training-file is not set")
	}
	if strings.Contains(err.Error(), "rle-name") || strings.Contains(err.Error(), "rle-version") {
		t.Fatalf("expected rle-name and rle-version to be optional, got %v", err)
	}
}

// stubbedTrain wires train to a transport that accepts an upload and a job
// creation, and returns the action alongside the buffer it writes to.
func stubbedTrain(t *testing.T, ctx context.Context, flags *rleTrainFlags) (*trainAction, *bytes.Buffer) {
	t.Helper()
	trainingFilePath := filepath.Join(t.TempDir(), "training.jsonl")
	if err := os.WriteFile(trainingFilePath, []byte("{\"input\":\"example\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(foundryProjectEndpointEnvVar, "https://account.services.ai.azure.com/api/projects/project")

	client := newFinetuneClientWithCredential("https://account.openai.azure.com", &testTokenCredential{})
	client.httpClient.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		// The upload streams through a pipe, so a transport that answers without
		// reading closes it under the writer.
		if request.Body != nil {
			if _, err := io.Copy(io.Discard, request.Body); err != nil {
				return nil, err
			}
		}
		body := `{"id":"file-training"}`
		if request.URL.Path == finetuneJobsPath {
			body = `{"id":"ftjob-1","status":"queued"}`
		}
		return &http.Response{
			StatusCode: http.StatusCreated,
			Body:       io.NopCloser(strings.NewReader(body)),
			Header:     make(http.Header),
		}, nil
	})

	originalCreateClient := createFinetuneClient
	createFinetuneClient = func(string) (*finetuneClient, error) { return client, nil }
	t.Cleanup(func() { createFinetuneClient = originalCreateClient })

	command := newTrainCommand()
	command.SetContext(ctx)
	output := &bytes.Buffer{}
	command.SetOut(output)
	command.SetErr(output)

	flags.rleName = "code_rl"
	flags.rleVersion = "1.0.0"
	flags.model = "Qwen/Qwen3-32B"
	flags.trainingFile = trainingFilePath
	return &trainAction{cmd: command, flags: flags}, output
}

// Without --follow the command exits, so there is no dashboard to serve. The
// rollout ids a run generates are not knowable ahead of time, so the job id is
// the only way back to them and has to be offered.
func TestTrainNamesTheMonitorCommandWhenItIsNotFollowing(t *testing.T) {
	action, output := stubbedTrain(t, context.Background(), &rleTrainFlags{})
	if err := action.Run(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "azd ai rle monitor --job-id ftjob-1") {
		t.Fatalf("output = %q, want the command that opens this run's rollouts", output.String())
	}
}

// An endpoint the run needed is an endpoint the monitor needs, so a pasted
// command has to carry it.
func TestTrainCarriesTheEndpointIntoTheMonitorCommand(t *testing.T) {
	action, output := stubbedTrain(t, context.Background(),
		&rleTrainFlags{endpoint: "https://account.openai.azure.com"})
	if err := action.Run(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "--job-id ftjob-1 --endpoint https://account.openai.azure.com") {
		t.Fatalf("output = %q, want the endpoint carried into the monitor command", output.String())
	}
}

// The point of following a run is watching its rollouts land, so the dashboard
// runs alongside the stream rather than after it, and outlives it: the run
// ending is when the rollouts are finally all there to read.
func TestTrainFollowServesTheRolloutDashboardUntilItIsStopped(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	started := make(chan string, 1)
	monitored := make(chan string, 1)
	originalMonitor := runJobMonitor
	runJobMonitor = func(
		ctx context.Context, _ rollouts.Reader, _ rollouts.Lister,
		jobID string, runDir string, _ bool, _, _ io.Writer,
	) error {
		started <- jobID
		monitored <- runDir
		<-ctx.Done()
		return nil
	}
	t.Cleanup(func() { runJobMonitor = originalMonitor })

	streamed := make(chan struct{})
	originalFollow := followTrainingRunFunc
	followTrainingRunFunc = func(
		context.Context, string, string, string, string, io.Writer,
	) (string, error) {
		close(streamed)
		return "succeeded", nil
	}
	t.Cleanup(func() { followTrainingRunFunc = originalFollow })

	logsRoot := t.TempDir()
	action, output := stubbedTrain(t, ctx,
		&rleTrainFlags{follow: true, noBrowser: true, logsRoot: logsRoot})

	returned := make(chan error, 1)
	go func() { returned <- action.Run() }()

	select {
	case jobID := <-started:
		if jobID != "ftjob-1" {
			t.Fatalf("dashboard opened %q, want the submitted job", jobID)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("--follow did not start the rollout dashboard")
	}

	// The dashboard must read the same directory the stream writes, or it would
	// show a run with no metrics while they are being downloaded beside it.
	if runDir := <-monitored; runDir != filepath.Join(logsRoot, "rle-harness", "ftjob-1") {
		t.Fatalf("dashboard read %q, want the run mirror --follow writes", runDir)
	}

	<-streamed
	// The run is over. The command must still be serving, or the window the user
	// was sent to would close on them.
	select {
	case err := <-returned:
		t.Fatalf("train returned %v when the run finished, want the dashboard held open", err)
	case <-time.After(250 * time.Millisecond):
	}

	cancel()
	select {
	case err := <-returned:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("train did not return after the dashboard was stopped")
	}

	if !strings.Contains(output.String(), "Final status: succeeded") {
		t.Fatalf("output = %q, want the run's final status", output.String())
	}
	if !strings.Contains(output.String(), "dashboard is still running") {
		t.Fatalf("output = %q, want the dashboard to outlive the run", output.String())
	}
}

func TestTrainingOptionsReachTheRleEnvironmentBlock(t *testing.T) {
	flags := &rleTrainFlags{rleName: "math_rl", rleVersion: "1.0.6", model: "qwen3-32b-1"}
	options := map[string]any{"learning_rate": 2e-5, "max_steps": float64(50)}

	request := buildFinetuneJobRequest(flags, "file-training", "", options)

	if request.Method.RleEnvironment.Hyperparameters["max_steps"] != float64(50) {
		t.Fatalf("expected max_steps to reach the request, got %#v",
			request.Method.RleEnvironment.Hyperparameters)
	}

	// The service reads options from method.rl_environment.hyperparameters, so a
	// rename in the payload would silently stop applying them.
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var wire struct {
		Method struct {
			RleEnvironment struct {
				Hyperparameters map[string]any `json:"hyperparameters"`
			} `json:"rl_environment"`
		} `json:"method"`
	}
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if wire.Method.RleEnvironment.Hyperparameters["learning_rate"] != 2e-5 {
		t.Fatalf("options did not survive the wire shape: %s", encoded)
	}
}

func TestNoTrainingOptionsLeavesTheRequestUnchanged(t *testing.T) {
	flags := &rleTrainFlags{rleName: "math_rl", rleVersion: "1.0.6", model: "qwen3-32b-1"}

	request := buildFinetuneJobRequest(flags, "file-training", "", nil)

	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(encoded), "hyperparameters") {
		t.Fatalf("absent options must not add the field: %s", encoded)
	}
}

// writeTrainRleConfigWithTrain writes a manifest carrying a [train] section.
func writeTrainRleConfigWithTrain(t *testing.T, dir string, train *project.RleTrainSettings) {
	t.Helper()
	schemaVersion := project.CurrentRleManifestSchemaVersion
	if err := project.WriteRleConfig(dir, project.RleConfig{
		SchemaVersion: &schemaVersion,
		Rle: project.RleManifest{
			Name:    "code_rl",
			Version: "1.2.3",
			Type:    project.RleTypeGym,
			Subtype: project.RleSubtypeOpenEnv,
		},
		Train: train,
	}); err != nil {
		t.Fatal(err)
	}
}

func strPtr(value string) *string { return &value }

func TestTrainReadsSettingsFromRleConfig(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	steps := 7
	writeTrainRleConfigWithTrain(t, dir, &project.RleTrainSettings{
		Model:           strPtr("qwen3-32b-1"),
		TrainingFile:    strPtr("job_data/train.jsonl"),
		ValidationFile:  strPtr("job_data/validation.jsonl"),
		Suffix:          strPtr("nightly"),
		MaxEpisodeSteps: &steps,
		Options:         map[string]any{"group_size": int64(4)},
	})

	action := &trainAction{flags: &rleTrainFlags{}}
	options, err := action.resolveTrainSettings()
	if err != nil {
		t.Fatal(err)
	}

	if action.flags.model != "qwen3-32b-1" {
		t.Fatalf("model: %q", action.flags.model)
	}
	if action.flags.trainingFile != "job_data/train.jsonl" {
		t.Fatalf("training file: %q", action.flags.trainingFile)
	}
	if action.flags.validationFile != "job_data/validation.jsonl" {
		t.Fatalf("validation file: %q", action.flags.validationFile)
	}
	if action.flags.suffix != "nightly" {
		t.Fatalf("suffix: %q", action.flags.suffix)
	}
	if action.flags.maxEpisodeSteps != 7 {
		t.Fatalf("max episode steps: %d", action.flags.maxEpisodeSteps)
	}
	if options["group_size"] != int64(4) {
		t.Fatalf("options: %#v", options)
	}
}

func TestTrainFlagsOverrideRleConfigTrainSection(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfigWithTrain(t, dir, &project.RleTrainSettings{
		Model:        strPtr("from-manifest"),
		TrainingFile: strPtr("manifest.jsonl"),
		Suffix:       strPtr("manifest"),
	})

	action := &trainAction{flags: &rleTrainFlags{
		model:        "from-flag",
		trainingFile: "flag.jsonl",
		suffix:       "flag",
	}}
	if _, err := action.resolveTrainSettings(); err != nil {
		t.Fatal(err)
	}

	if action.flags.model != "from-flag" || action.flags.trainingFile != "flag.jsonl" {
		t.Fatalf("flags must win: %q %q", action.flags.model, action.flags.trainingFile)
	}
	if action.flags.suffix != "flag" {
		t.Fatalf("suffix: %q", action.flags.suffix)
	}
}

func TestTaskCountSetsTheDatasetLimitOption(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfigWithTrain(t, dir, &project.RleTrainSettings{
		Model:        strPtr("qwen3-32b-1"),
		TrainingFile: strPtr("train.jsonl"),
	})

	action := &trainAction{flags: &rleTrainFlags{taskCount: 12}}
	options, err := action.resolveTrainSettings()
	if err != nil {
		t.Fatal(err)
	}
	if options[trainTaskCountOption] != 12 {
		t.Fatalf("expected --task-count to set %s, got %#v", trainTaskCountOption, options)
	}
}

func TestTaskCountOverridesTheManifestLimit(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfigWithTrain(t, dir, &project.RleTrainSettings{
		Model:        strPtr("qwen3-32b-1"),
		TrainingFile: strPtr("train.jsonl"),
		Options:      map[string]any{trainTaskCountOption: int64(500)},
	})

	action := &trainAction{flags: &rleTrainFlags{taskCount: 12}}
	options, err := action.resolveTrainSettings()
	if err != nil {
		t.Fatal(err)
	}
	// One option cannot be sent twice, so the flag has to replace the manifest's
	// value rather than being appended alongside it.
	if options[trainTaskCountOption] != 12 {
		t.Fatalf("expected the flag to win, got %#v", options[trainTaskCountOption])
	}
}

func TestTaskCountIsOmittedWhenUnset(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfigWithTrain(t, dir, &project.RleTrainSettings{
		Model:        strPtr("qwen3-32b-1"),
		TrainingFile: strPtr("train.jsonl"),
	})

	action := &trainAction{flags: &rleTrainFlags{}}
	options, err := action.resolveTrainSettings()
	if err != nil {
		t.Fatal(err)
	}
	if _, present := options[trainTaskCountOption]; present {
		t.Fatalf("an unset --task-count must not limit the dataset: %#v", options)
	}
}

func TestTrainReportsWhatIsMissingWithoutFlagsOrManifestSettings(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfig(t, dir, "code_rl", "1.2.3")

	action := &trainAction{flags: &rleTrainFlags{}}
	_, err := action.resolveTrainSettings()
	if err == nil {
		t.Fatal("expected a missing model to be reported")
	}
	var localErr *azdext.LocalError
	if !errors.As(err, &localErr) || localErr.Code != "rle_train_setting_required" {
		t.Fatalf("expected a user-facing error, got %#v", err)
	}
	if !strings.Contains(localErr.Suggestion, "rle.toml") {
		t.Fatalf("the suggestion should point at rle.toml: %q", localErr.Suggestion)
	}
}

// A [train] section must not be mistaken for the published defaults: publish sends
// Defaults to the service and rejects a local copy that has drifted, so a per-run
// value landing there would make every tweak look like a mismatch.
func TestTrainSectionIsSeparateFromPublishedDefaults(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeTrainRleConfigWithTrain(t, dir, &project.RleTrainSettings{
		Model:        strPtr("qwen3-32b-1"),
		TrainingFile: strPtr("train.jsonl"),
		Options:      map[string]any{"group_size": int64(4)},
	})

	config, err := project.LoadRleConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if config.Defaults != nil {
		t.Fatalf("[train] must not populate the published defaults: %#v", config.Defaults)
	}
	if config.Train == nil || config.Train.Model == nil {
		t.Fatal("expected the train section to round-trip through rle.toml")
	}
}
