// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"fmt"
	"net/url"
	"os"
	"strings"

	"github.com/azure/azure-dev/cli/azd/pkg/azdext"
)

// RLE fine-tuning jobs are not served by a Foundry account's own fine-tuning
// API yet, so deriving the endpoint from the project reaches a service that
// cannot accept them. Point at the deployment that runs them instead.
//
// This is deliberately temporary. When an account's endpoint serves RLE jobs,
// delete this and let resolveFinetuneEndpoint fall back to the project again --
// finetuneEndpointFromFoundryProject is still here and still tested for that.
const rleTrainingServiceEndpoint = "https://rle-train-facade.ambitiouswater-3eb7bce4.eastus2.azurecontainerapps.io"

// rleTrainEndpointEnvVar names a different deployment without rebuilding, so a
// second environment is not blocked on changing the constant above.
const rleTrainEndpointEnvVar = "RLE_TRAIN_ENDPOINT"

// resolveFinetuneEndpoint picks the fine-tuning API the RLE commands talk to:
// the --endpoint flag, else RLE_TRAIN_ENDPOINT, else the RLE training service.
// All three RLE commands share it so that a job submitted by train is the one
// jobs and monitor look for.
func resolveFinetuneEndpoint(flagValue string, projectEndpoint string) (string, error) {
	if raw := strings.TrimSpace(flagValue); raw != "" {
		return normalizeFinetuneEndpoint(raw)
	}
	if raw := strings.TrimSpace(os.Getenv(rleTrainEndpointEnvVar)); raw != "" {
		return normalizeFinetuneEndpoint(raw)
	}
	return normalizeFinetuneEndpoint(rleTrainingServiceEndpoint)
}

func finetuneEndpointFromFoundryProject(projectEndpoint string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(projectEndpoint))
	if err != nil {
		return "", invalidProjectEndpointError(fmt.Sprintf("invalid Foundry project endpoint: %v", err))
	}

	const foundryHostSuffix = ".services.ai.azure.com"
	host := strings.ToLower(u.Hostname())
	account := strings.TrimSuffix(host, foundryHostSuffix)
	if account == "" || account == host {
		return "", invalidProjectEndpointError(
			"Foundry project endpoint host must end with .services.ai.azure.com",
		)
	}

	return (&url.URL{
		Scheme: "https",
		Host:   account + ".openai.azure.com",
	}).String(), nil
}

func normalizeFinetuneEndpoint(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", invalidFinetuneEndpointError(fmt.Sprintf("invalid fine-tuning API endpoint: %v", err))
	}
	if !strings.EqualFold(u.Scheme, "https") {
		return "", invalidFinetuneEndpointError("fine-tuning API endpoint must use https")
	}
	if u.Hostname() == "" {
		return "", invalidFinetuneEndpointError("fine-tuning API endpoint must include a host")
	}

	u.Scheme = "https"
	u.Host = strings.ToLower(u.Host)
	u.Path = strings.TrimRight(u.Path, "/")
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

func invalidFinetuneEndpointError(message string) error {
	return &azdext.LocalError{
		Message:    message,
		Code:       "rle_invalid_train_endpoint",
		Category:   azdext.LocalErrorCategoryUser,
		Suggestion: "Pass --endpoint https://<resource>.openai.azure.com.",
	}
}
