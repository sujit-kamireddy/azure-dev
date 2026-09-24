// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
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
	taskCount       int
	endpoint        string

	// Where the merged settings came from, for the line printed before submitting.
	optionsSource string
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
	cmd.Flags().StringVar(&flags.model, "model", "", "Base model id to fine-tune. Defaults to train.model in ./rle.toml.")
	cmd.Flags().StringVar(&flags.trainingFile, "training-file", "",
		"Path to the local training dataset uploaded as the Loom job input. "+
			"Defaults to train.training_file in ./rle.toml.")
	cmd.Flags().StringVar(&flags.validationFile, "validation-file", "",
		"Path to a local validation dataset to upload. Defaults to train.validation_file in ./rle.toml.")
	cmd.Flags().StringVar(&flags.suffix, "suffix", "",
		"Suffix appended to the resulting fine-tuned model name. Defaults to train.suffix in ./rle.toml.")
	cmd.Flags().IntVar(&flags.maxEpisodeSteps, "max-episode-steps", 0,
		"Maximum steps the RLE executes per rollout (0 uses the service default).")
	cmd.Flags().IntVar(&flags.taskCount, "task-count", 0,
		"Train on only the first N tasks of the training dataset, for a smaller run. "+
			"Sets the max_train_examples training option (0 uses the whole dataset).")
	cmd.Flags().StringVar(&flags.endpoint, "endpoint", "",
		fmt.Sprintf("Fine-tuning API endpoint. Defaults to the account in %s.", foundryProjectEndpointEnvVar))

	// model and training-file are not marked required: rle.toml can supply either,
	// and cobra would reject the run before the manifest is ever read.

	return cmd
}

// resolveTrainSettings fills in everything train can take from rle.toml in the
// current folder: the RLE name and version, and the [train] section's model,
// dataset paths, suffix and options. Flags win over the manifest, so a saved
// configuration stays overridable for one run without editing it.
//
// The manifest is optional. It is only required when a flag left something
// unresolved, so `train --rle-name ... --model ... --training-file ...` still
// works from a folder that has no rle.toml.
func (a *trainAction) resolveTrainSettings() (map[string]any, error) {
	name := strings.TrimSpace(a.flags.rleName)
	version := strings.TrimSpace(a.flags.rleVersion)
	model := strings.TrimSpace(a.flags.model)
	trainingFile := strings.TrimSpace(a.flags.trainingFile)
	validationFile := strings.TrimSpace(a.flags.validationFile)
	suffix := strings.TrimSpace(a.flags.suffix)

	options := map[string]any{}
	config, configErr := project.LoadRleConfig(".")
	if configErr == nil {
		if name == "" {
			name = config.Rle.Name
		}
		if version == "" {
			version = config.Rle.Version
		}
		if train := config.Train; train != nil {
			if model == "" && train.Model != nil {
				model = strings.TrimSpace(*train.Model)
			}
			if trainingFile == "" && train.TrainingFile != nil {
				trainingFile = strings.TrimSpace(*train.TrainingFile)
			}
			if validationFile == "" && train.ValidationFile != nil {
				validationFile = strings.TrimSpace(*train.ValidationFile)
			}
			if suffix == "" && train.Suffix != nil {
				suffix = strings.TrimSpace(*train.Suffix)
			}
			if a.flags.maxEpisodeSteps == 0 && train.MaxEpisodeSteps != nil {
				a.flags.maxEpisodeSteps = *train.MaxEpisodeSteps
			}
			for optionName, value := range train.Options {
				options[optionName] = value
			}
			if len(options) > 0 {
				a.flags.optionsSource = project.RleConfigFile
			}
		}
	} else if name == "" || version == "" {
		// Only the environment identity structurally needs the manifest. A missing
		// model or dataset is reported by name below, which is more useful than
		// naming a file the caller may not have intended to use.
		return nil, configErr
	}

	// --task-count is shorthand for one training option, so it is applied the way
	// any other flag overrides the manifest rather than being sent separately.
	if a.flags.taskCount > 0 {
		options[trainTaskCountOption] = a.flags.taskCount
		if a.flags.optionsSource == "" {
			a.flags.optionsSource = "--task-count"
		} else {
			a.flags.optionsSource = project.RleConfigFile + " and --task-count"
		}
	}

	if model == "" {
		return nil, missingTrainSettingError("model", "--model", "train.model")
	}
	if trainingFile == "" {
		return nil, missingTrainSettingError("training file", "--training-file", "train.training_file")
	}

	normalized, err := project.NormalizeRleVersion(version)
	if err != nil {
		return nil, err
	}
	a.flags.rleName = name
	a.flags.rleVersion = normalized
	a.flags.model = model
	a.flags.trainingFile = trainingFile
	a.flags.validationFile = validationFile
	a.flags.suffix = suffix
	return options, nil
}

// trainTaskCountOption is the training option --task-count sets. The service
// applies it as the dataset limit for the run; the CLI only forwards it.
const trainTaskCountOption = "max_train_examples"

func missingTrainSettingError(what string, flag string, manifestKey string) error {
	return &azdext.LocalError{
		Message:  fmt.Sprintf("A %s is required for train.", what),
		Code:     "rle_train_setting_required",
		Category: azdext.LocalErrorCategoryUser,
		Suggestion: fmt.Sprintf(
			"Pass %s, or set %s in %s.", flag, manifestKey, project.RleConfigFile,
		),
	}
}

func (a *trainAction) Run() error {
	trainingOptions, err := a.resolveTrainSettings()
	if err != nil {
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

	request := buildFinetuneJobRequest(a.flags, trainingFileID, validationFileID, trainingOptions)

	if _, err := fmt.Fprintf(
		a.cmd.OutOrStdout(),
		"Submitting rl_environment fine-tuning job for RLE '%s' version %s (model=%s) ...\n",
		a.flags.rleName,
		a.flags.rleVersion,
		a.flags.model,
	); err != nil {
		return err
	}

	if len(trainingOptions) > 0 {
		names := make([]string, 0, len(trainingOptions))
		for name := range trainingOptions {
			names = append(names, name)
		}
		sort.Strings(names)
		if _, err := fmt.Fprintf(
			a.cmd.OutOrStdout(),
			"Applying %d training option(s) from %s: %s\n",
			len(names),
			a.flags.optionsSource,
			strings.Join(names, ", "),
		); err != nil {
			return err
		}
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
	options map[string]any,
) finetuneJobCreationRequest {
	rleEnvironment := finetuneRleEnvironmentConfig{
		Name:    flags.rleName,
		Version: flags.rleVersion,
	}
	if flags.maxEpisodeSteps > 0 {
		steps := flags.maxEpisodeSteps
		rleEnvironment.MaxEpisodeSteps = &steps
	}
	if len(options) > 0 {
		rleEnvironment.Hyperparameters = options
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
