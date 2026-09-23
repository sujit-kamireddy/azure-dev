// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"encoding/json"
	"testing"
)

func TestLoomOperationEnvelopeAcceptsStringAndObjectErrors(t *testing.T) {
	for _, tc := range []struct {
		name    string
		payload string
		message string
		code    string
	}{
		{
			name:    "string",
			payload: `{"status":"failed","error":"The model allocation is unavailable."}`,
			message: "The model allocation is unavailable.",
		},
		{
			name:    "object",
			payload: `{"status":"failed","error":{"code":"CapacityUnavailable","message":"Try another model."}}`,
			message: "Try another model.",
			code:    "CapacityUnavailable",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var envelope loomOperationEnvelope
			if err := json.Unmarshal([]byte(tc.payload), &envelope); err != nil {
				t.Fatal(err)
			}
			if envelope.Error == nil ||
				envelope.Error.Message != tc.message ||
				envelope.Error.Code != tc.code {
				t.Fatalf("unexpected Loom error: %#v", envelope.Error)
			}
		})
	}
}
