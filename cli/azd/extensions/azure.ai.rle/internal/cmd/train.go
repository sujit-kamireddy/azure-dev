// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"azure.ai.rle/internal/project"
	"github.com/azure/azure-dev/cli/azd/pkg/azdext"
	"github.com/spf13/cobra"
)

type rleTrainFlags struct {
	rleName         string
	rleVersion      string
	model           string
	trainingFile    string
	validationFile  string
	suffix          string
	maxEpisodeSteps int
	follow          bool
	logsRoot        string
	noBrowser       bool
	endpoint        string
}

type trainAction struct {
	cmd   *cobra.Command
	flags *rleTrainFlags
}

// newTrainCommand submits a reinforcement fine-tuning job that uses a published RLE
// environment as the reward source (finetunesapi method type rl_environment) instead of
// a grader. This method is currently hidden from finetunesapi's public Swagger surface
// and only completes for base models the service has enabled for Loom-backed
// RL-environment training, so it is gated behind AZD_AI_RLE_ENABLE_ALL for the RLE
// team's own iteration.
func newTrainCommand() *cobra.Command {
	flags := &rleTrainFlags{}

	cmd := &cobra.Command{
		Use:   "train",
		Short: "Submit an RLE-backed reinforcement fine-tuning job (experimental)",
		Long: `Submit an RLE-backed reinforcement fine-tuning job (experimental).

This uses finetunesapi's rl_environment fine-tuning method: the named, published RLE
environment supplies the reward signal instead of a grader. The command uploads the local
training file to the fine-tuning resource before Loom mounts it as the job input. rl_environment
is currently hidden from finetunesapi's public API surface and only completes for base models
enabled for Loom-backed RL-environment training. Job creation fails if the base model is not
enabled, or if the RLE version is not published and ready in the project set by
FOUNDRY_PROJECT_ENDPOINT.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return (&trainAction{cmd: cmd, flags: flags}).Run()
		},
	}

	cmd.Flags().StringVar(&flags.rleName, "rle-name", "",
		"Name of the published RLE environment to train against. Defaults to rle.name in ./rle.toml.")
	cmd.Flags().StringVar(&flags.rleVersion, "rle-version", "",
		"Version of the published RLE environment. Defaults to rle.version in ./rle.toml.")
	cmd.Flags().StringVar(&flags.model, "model", "", "Base model id to fine-tune.")
	cmd.Flags().StringVar(&flags.trainingFile, "training-file", "",
		"Path to the local training dataset uploaded as the Loom job input.")
	cmd.Flags().StringVar(&flags.validationFile, "validation-file", "",
		"Path to a local validation dataset to upload.")
	cmd.Flags().StringVar(&flags.suffix, "suffix", "", "Suffix appended to the resulting fine-tuned model name.")
	cmd.Flags().IntVar(&flags.maxEpisodeSteps, "max-episode-steps", 0,
		"Maximum steps the RLE executes per rollout (0 uses the service default).")
	cmd.Flags().BoolVar(&flags.follow, "follow", false,
		"Stream the run's logs and metrics locally until the job finishes, in the layout "+
			"the Loom dashboard reads, and serve the run's rollouts in a local dashboard.")
	cmd.Flags().StringVar(&flags.logsRoot, "logs-root", "",
		"Where --follow writes mirrored runs. Defaults to $LOOM_LOGS_ROOT, else ~/loom-runs.")
	cmd.Flags().BoolVar(&flags.noBrowser, "no-browser", false,
		"With --follow, print the rollout dashboard address without opening a browser.")
	cmd.Flags().StringVar(&flags.endpoint, "endpoint", "",
		fmt.Sprintf("Fine-tuning API endpoint. Defaults to the account in %s.", foundryProjectEndpointEnvVar))

	for _, name := range []string{"model", "training-file"} {
		_ = cmd.MarkFlagRequired(name)
	}

	return cmd
}

// resolveTrainTarget fills in the RLE name and version from rle.toml in the current
// folder whenever either is omitted, so train can be run from an environment folder
// the same way rollout can.
func (a *trainAction) resolveTrainTarget() error {
	name := strings.TrimSpace(a.flags.rleName)
	version := strings.TrimSpace(a.flags.rleVersion)
	if name == "" || version == "" {
		config, err := project.LoadRleConfig(".")
		if err != nil {
			return err
		}
		if name == "" {
			name = config.Rle.Name
		}
		if version == "" {
			version = config.Rle.Version
		}
	}
	normalized, err := project.NormalizeRleVersion(version)
	if err != nil {
		return err
	}
	a.flags.rleName = name
	a.flags.rleVersion = normalized
	return nil
}

func (a *trainAction) Run() error {
	if err := a.resolveTrainTarget(); err != nil {
		return err
	}

	trainingFilePath, err := resolveLocalFilePath(a.flags.trainingFile, "training", true)
	if err != nil {
		return err
	}
	validationFilePath, err := resolveLocalFilePath(a.flags.validationFile, "validation", false)
	if err != nil {
		return err
	}

	projectEndpoint, err := resolveFoundryProjectEndpoint()
	if err != nil {
		return err
	}
	if projectEndpoint == "" {
		return &azdext.LocalError{
			Message:  "Foundry project endpoint is required for train.",
			Code:     "rle_project_required",
			Category: azdext.LocalErrorCategoryUser,
			Suggestion: fmt.Sprintf(
				"Set %s=https://<account>.services.ai.azure.com/api/projects/<project>.",
				foundryProjectEndpointEnvVar,
			),
		}
	}
	endpoint, err := resolveFinetuneEndpoint(a.flags.endpoint, projectEndpoint)
	if err != nil {
		return err
	}
	azureAIProject, err := projectRouteSegment(projectEndpoint)
	if err != nil {
		return err
	}

	client, err := createFinetuneClient(endpoint)
	if err != nil {
		return err
	}

	trainingFileID, err := a.uploadInputFile(client, trainingFilePath, "training")
	if err != nil {
		return err
	}
	validationFileID := ""
	if validationFilePath != "" {
		validationFileID, err = a.uploadInputFile(client, validationFilePath, "validation")
		if err != nil {
			return err
		}
	}

	request := buildFinetuneJobRequest(a.flags, trainingFileID, validationFileID)

	if _, err := fmt.Fprintf(
		a.cmd.OutOrStdout(),
		"Submitting rl_environment fine-tuning job for RLE '%s' version %s (model=%s) ...\n",
		a.flags.rleName,
		a.flags.rleVersion,
		a.flags.model,
	); err != nil {
		return err
	}

	job, err := client.createJob(a.cmd.Context(), request, azureAIProject)
	if err != nil {
		return finetuneServiceError(err)
	}

	if _, err := fmt.Fprintf(
		a.cmd.OutOrStdout(),
		"\nSubmitted fine-tuning job %s (status=%s).\n",
		job.Id,
		job.Status,
	); err != nil {
		return err
	}

	body, err := json.MarshalIndent(job, "", "  ")
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintln(a.cmd.OutOrStdout(), string(body)); err != nil {
		return err
	}

	if a.flags.follow {
		return a.followJob(client, job.Id)
	}

	// Without --follow the command is about to exit, so there is nothing to
	// serve a dashboard from. Name the command that opens one instead: the
	// rollout ids a run generates are not knowable ahead of time, so the job id
	// is the only way back to them.
	monitorEndpoint := strings.TrimSpace(a.flags.endpoint)
	if monitorEndpoint != "" {
		monitorEndpoint = fmt.Sprintf(" --endpoint %s", monitorEndpoint)
	}
	if _, err := fmt.Fprintf(
		a.cmd.OutOrStdout(),
		"\nWatch this run's rollouts as they land:\n  azd ai rle monitor --job-id %s%s\n",
		job.Id,
		monitorEndpoint,
	); err != nil {
		return err
	}
	return nil
}

// followJob mirrors the run's artifacts locally and serves its rollouts, until
// the run finishes and the user stops watching.
//
// A streaming failure is reported but does not fail the command: the job was
// accepted and is running on the service, and exiting non-zero would suggest it
// was not. The job id is printed above, so it stays recoverable.
func (a *trainAction) followJob(client *finetuneClient, jobID string) error {
	authorization, err := client.authorizationHeader(a.cmd.Context())
	if err != nil {
		return fmt.Errorf("acquire a token for the run stream: %w", err)
	}

	logsRoot := strings.TrimSpace(a.flags.logsRoot)
	if logsRoot == "" {
		logsRoot = defaultLogsRoot()
	}

	// The dashboard runs alongside the stream rather than after it. A run
	// records rollouts for as long as it lasts, and the point of following one
	// is to watch them land, not to read them once it is over.
	//
	// It starts empty: the first rollout of a run takes minutes. The page polls,
	// so rollouts appear as the run records them.
	dashboard := make(chan struct{})
	go func() {
		defer close(dashboard)
		source := &jobRollouts{client: client, jobID: jobID}
		if err := runJobMonitor(
			a.cmd.Context(), source, source, jobID, a.flags.noBrowser,
			a.cmd.OutOrStdout(), a.cmd.ErrOrStderr(),
		); err != nil {
			// The stream is the part that must keep working; a dashboard that
			// cannot start is worth saying once and no more.
			fmt.Fprintf(a.cmd.ErrOrStderr(), "The rollout dashboard did not start: %v\n", err)
		}
	}()

	status, streamErr := followTrainingRunFunc(
		a.cmd.Context(),
		client.baseUrl,
		authorization,
		jobID,
		logsRoot,
		a.cmd.OutOrStdout(),
	)
	if streamErr != nil {
		fmt.Fprintf(a.cmd.ErrOrStderr(),
			"\nStopped following %s: %v\nThe job is still running on the service; "+
				"check it with: azd ai rle jobs\n", jobID, streamErr)
	} else if status != "" {
		fmt.Fprintf(a.cmd.OutOrStdout(), "Final status: %s\n", status)
	}

	// The run is over but its rollouts are not read yet. Hold the dashboard open
	// until the user stops it, rather than closing the window they were sent to.
	fmt.Fprintf(a.cmd.OutOrStdout(),
		"\nThe rollout dashboard is still running. Press Ctrl+C to stop it.\n")
	<-dashboard
	return nil
}

func (a *trainAction) uploadInputFile(client *finetuneClient, filePath string, fileType string) (string, error) {
	if _, err := fmt.Fprintf(a.cmd.OutOrStdout(), "Uploading %s file %q ...\n", fileType, filepath.Base(filePath)); err != nil {
		return "", err
	}

	uploadedFile, err := client.uploadFile(a.cmd.Context(), filePath)
	if err != nil {
		return "", finetuneUploadServiceError(err)
	}

	if _, err := fmt.Fprintf(a.cmd.OutOrStdout(), "Uploaded %s file as %s.\n", fileType, uploadedFile.Id); err != nil {
		return "", err
	}
	return uploadedFile.Id, nil
}

func resolveLocalFilePath(raw string, fileType string, required bool) (string, error) {
	filePath := strings.TrimSpace(raw)
	if filePath == "" {
		if !required {
			return "", nil
		}
		return "", &azdext.LocalError{
			Message:    fmt.Sprintf("A local %s file path is required for train.", fileType),
			Code:       "rle_train_training_file_required",
			Category:   azdext.LocalErrorCategoryUser,
			Suggestion: fmt.Sprintf("Pass the path to a local %s dataset using --%s-file.", fileType, fileType),
		}
	}

	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return "", &azdext.LocalError{
			Message:    fmt.Sprintf("Unable to access the %s file.", fileType),
			Code:       "rle_train_file_unavailable",
			Category:   azdext.LocalErrorCategoryUser,
			Suggestion: fmt.Sprintf("Verify that --%s-file points to a readable local file.", fileType),
		}
	}
	if !fileInfo.Mode().IsRegular() {
		return "", &azdext.LocalError{
			Message:    fmt.Sprintf("The %s file must be a regular file.", fileType),
			Code:       "rle_train_file_not_regular",
			Category:   azdext.LocalErrorCategoryUser,
			Suggestion: fmt.Sprintf("Pass the path to a local %s dataset file using --%s-file.", fileType, fileType),
		}
	}
	return filePath, nil
}

func buildFinetuneJobRequest(
	flags *rleTrainFlags,
	trainingFileID string,
	validationFileID string,
) finetuneJobCreationRequest {
	rleEnvironment := finetuneRleEnvironmentConfig{
		Name:    flags.rleName,
		Version: flags.rleVersion,
	}
	if flags.maxEpisodeSteps > 0 {
		steps := flags.maxEpisodeSteps
		rleEnvironment.MaxEpisodeSteps = &steps
	}

	request := finetuneJobCreationRequest{
		Model:        flags.model,
		TrainingFile: trainingFileID,
		TrainingType: finetuneTrainingTypeGlobalStandard,
		Method: &finetuneMethodRequest{
			Type:           finetuneMethodTypeRleEnvironment,
			RleEnvironment: rleEnvironment,
		},
	}
	if validationFileID != "" {
		request.ValidationFile = &validationFileID
	}
	if flags.suffix != "" {
		request.Suffix = &flags.suffix
	}
	return request
}
