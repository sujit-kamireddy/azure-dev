// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"azure.ai.rle/internal/monitor"
	"azure.ai.rle/internal/rollouts"

	"github.com/azure/azure-dev/cli/azd/pkg/azdext"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

var runRolloutMonitor = monitor.Run
var runJobMonitor = monitor.RunJob

func newMonitorCommand() *cobra.Command {
	var rolloutID string
	var jobID string
	var endpoint string
	var noBrowser bool
	var outputDir string
	cmd := &cobra.Command{
		Use:   "monitor (--rollout-id <id> | --job-id <id>)",
		Short: "Open a local dashboard for a saved rollout or a training job's rollouts",
		Long: `Open a read-only browser dashboard for saved rollouts.

With --rollout-id, read .output/<rollout-id>/summary.json and rollout.json, or
use --output-dir to select the artifact root used during execution. No sign-in,
Foundry project setting, local source folder, or running sandbox is required.
If the rollout directory does not exist, the command warns and exits without
opening a browser. Incomplete or corrupt artifacts still return an error.

With --job-id, list the rollouts a training job recorded and open any one of
them in the same dashboard. This reads the fine-tuning service, so it requires
sign-in and an endpoint; --output-dir is not used. Rollout bodies are fetched
only as they are opened.

The monitor stays running until Ctrl+C. Use --no-browser to open the printed
link manually and enter the local access code.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			rolloutID = strings.TrimSpace(rolloutID)
			jobID = strings.TrimSpace(jobID)
			if rolloutID != "" && jobID != "" {
				return &azdext.LocalError{
					Message:    "--rollout-id and --job-id cannot be used together.",
					Code:       "rle_monitor_conflicting_arguments",
					Category:   azdext.LocalErrorCategoryUser,
					Suggestion: "Pass --job-id to browse a training run, or --rollout-id to open one saved rollout.",
				}
			}
			if rolloutID == "" && jobID == "" {
				return &azdext.LocalError{
					Message:    "--rollout-id or --job-id is required but neither was provided.",
					Code:       "rle_monitor_rollout_id_required",
					Category:   azdext.LocalErrorCategoryUser,
					Suggestion: "Provide --rollout-id <id> from azd ai rle rollout, or --job-id <id> from azd ai rle train.",
				}
			}
			if err := validateMonitorOutput(cmd); err != nil {
				return err
			}
			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
			defer stop()
			if jobID != "" {
				return runMonitorForJob(ctx, cmd, jobID, endpoint, noBrowser)
			}
			if err := rollouts.ValidateID(rolloutID); err != nil {
				return invalidMonitorIDError(err)
			}
			directory, err := resolveRolloutOutputDir(outputDir)
			if err != nil {
				return err
			}
			reader := &rollouts.ArtifactReader{OutputDir: directory}
			err = runRolloutMonitor(ctx, reader, rolloutID, noBrowser, cmd.OutOrStdout(), cmd.ErrOrStderr())
			if errors.Is(err, rollouts.ErrArtifactDirectoryNotFound) {
				_, writeErr := fmt.Fprintf(cmd.ErrOrStderr(),
					"Warning: no output directory exists for rollout %s at %s.\n"+
						"No dashboard was opened. Older rollouts may not have exported artifacts; "+
						"check --output-dir if they were saved elsewhere.\n",
					rolloutID, filepath.Join(directory, rolloutID))
				return writeErr
			}
			return err
		},
	}
	cmd.Flags().StringVar(&rolloutID, "rollout-id", "", "ID of a rollout in the artifact directory.")
	cmd.Flags().StringVar(&jobID, "job-id", "", "ID of a training job whose recorded rollouts to browse.")
	cmd.Flags().StringVar(&endpoint, "endpoint", "", "Fine-tuning endpoint that owns the job (used with --job-id).")
	cmd.Flags().StringVar(&outputDir, "output-dir", defaultRolloutOutputDir, "Artifact root used by rollout --output-dir.")
	cmd.Flags().BoolVar(&noBrowser, "no-browser", false, "Print the dashboard link without opening a browser.")
	return cmd
}

// jobRollouts adapts the fine-tuning client to the dashboard's reader and lister.
//
// The service returns rollouts in the dashboard's own snapshot shape, so a
// recorded training rollout renders through exactly the same path as a local one.
type jobRollouts struct {
	client *finetuneClient
	jobID  string
}

func (j *jobRollouts) Get(ctx context.Context, rolloutID string) (rollouts.Snapshot, error) {
	snapshot, err := j.client.getJobRollout(ctx, j.jobID, rolloutID)
	if err != nil {
		return rollouts.Snapshot{}, err
	}
	return *snapshot, nil
}

// List follows the service's paging so the dashboard sees the whole run, not its first page.
func (j *jobRollouts) List(ctx context.Context) ([]rollouts.Entry, error) {
	const pageSize = 500
	var all []rollouts.Entry
	after := ""
	for {
		page, err := j.client.listJobRollouts(ctx, j.jobID, "", after, pageSize)
		if err != nil {
			return nil, err
		}
		all = append(all, page.Data...)
		if !page.HasMore || len(page.Data) == 0 {
			return all, nil
		}
		after = page.Data[len(page.Data)-1].RolloutID
	}
}

func runMonitorForJob(
	ctx context.Context,
	cmd *cobra.Command,
	jobID string,
	endpoint string,
	noBrowser bool,
) error {
	// Only the fine-tuning endpoint matters here; skip the project lookup when it is given.
	projectEndpoint := ""
	if strings.TrimSpace(endpoint) == "" {
		resolved, err := resolveJobsProjectEndpoint()
		if err != nil {
			return err
		}
		projectEndpoint = resolved
	}
	finetuneEndpoint, err := resolveFinetuneEndpoint(endpoint, projectEndpoint)
	if err != nil {
		return err
	}
	client, err := newFinetuneClient(finetuneEndpoint)
	if err != nil {
		return err
	}
	source := &jobRollouts{client: client, jobID: jobID}
	return runJobMonitor(ctx, source, source, jobID, noBrowser, cmd.OutOrStdout(), cmd.ErrOrStderr())
}

func resolveRolloutOutputDir(directory string) (string, error) {
	if strings.TrimSpace(directory) == "" {
		return "", &azdext.LocalError{
			Message: "--output-dir requires a non-empty directory.", Code: "rle_rollout_output_required",
			Category: azdext.LocalErrorCategoryUser, Suggestion: "Use .output or the artifact root used during execution.",
		}
	}
	path, err := filepath.Abs(directory)
	if err != nil {
		return "", fmt.Errorf("resolve rollout artifact directory: %w", err)
	}
	return path, nil
}

// monitorIsInteractive reports whether the dashboard has someone to show itself
// to. The monitor serves until Ctrl+C, so opening it for a script or a CI job
// would hang a rollout that has already done its work. A redirected stdout is
// the signal that nobody is watching.
var monitorIsInteractive = func(out io.Writer) bool {
	file, ok := out.(*os.File)
	return ok && term.IsTerminal(int(file.Fd()))
}

func invalidMonitorIDError(err error) error {
	return &azdext.LocalError{
		Message:    err.Error(),
		Code:       "rle_invalid_rollout_id",
		Category:   azdext.LocalErrorCategoryUser,
		Suggestion: "Use the 32-character rollout ID printed by azd ai rle rollout.",
	}
}

func validateMonitorOutput(cmd *cobra.Command) error {
	flag := cmd.Flag("output")
	if flag != nil && flag.Changed {
		return &azdext.LocalError{
			Message:    "--output cannot be used with the rollout monitor.",
			Code:       "rle_monitor_conflicting_arguments",
			Category:   azdext.LocalErrorCategoryUser,
			Suggestion: "Remove --output. The monitor prints its local browser link and access code.",
		}
	}
	return nil
}
