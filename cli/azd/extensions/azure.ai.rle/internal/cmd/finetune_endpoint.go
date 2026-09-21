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

// finetuneEndpointEnvVar optionally overrides the Azure OpenAI resource that hosts the
// fine-tuning API (POST /openai/v1/fine_tuning/jobs), e.g. https://<resource>.openai.azure.com.
// When it is unset, train derives that resource endpoint from FOUNDRY_PROJECT_ENDPOINT.
const finetuneEndpointEnvVar = "AZD_AI_RLE_TRAIN_ENDPOINT"

func resolveFinetuneEndpoint(flagValue string, projectEndpoint string) (string, error) {
	raw := strings.TrimSpace(flagValue)
	if raw == "" {
		raw = strings.TrimSpace(os.Getenv(finetuneEndpointEnvVar))
	}
	if raw != "" {
		return normalizeFinetuneEndpoint(raw)
	}
	if strings.TrimSpace(projectEndpoint) == "" {
		return "", &azdext.LocalError{
			Message:  "A fine-tuning API endpoint is required for train.",
			Code:     "rle_train_endpoint_required",
			Category: azdext.LocalErrorCategoryUser,
			Suggestion: fmt.Sprintf(
				"Set %s=https://<resource>.openai.azure.com, or pass --endpoint.",
				finetuneEndpointEnvVar,
			),
		}
	}

	return finetuneEndpointFromProject(projectEndpoint)
}

func finetuneEndpointFromProject(projectEndpoint string) (string, error) {
	normalizedProjectEndpoint, err := normalizeFoundryProjectEndpoint(projectEndpoint)
	if err != nil {
		return "", err
	}

	projectURL, err := url.Parse(normalizedProjectEndpoint)
	if err != nil {
		return "", invalidFinetuneEndpointError(fmt.Sprintf("invalid Foundry project endpoint: %v", err))
	}
	resourceName := strings.TrimSuffix(projectURL.Hostname(), ".services.ai.azure.com")
	if resourceName == "" {
		return "", invalidFinetuneEndpointError("Foundry project endpoint must include an Azure AI resource name")
	}

	return normalizeFinetuneEndpoint("https://" + resourceName + ".openai.azure.com")
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
		Message:  message,
		Code:     "rle_invalid_train_endpoint",
		Category: azdext.LocalErrorCategoryUser,
		Suggestion: fmt.Sprintf(
			"Set %s=https://<resource>.openai.azure.com.",
			finetuneEndpointEnvVar,
		),
	}
}
