# Prompt-injection fixtures (Chapter 26)

Inert test documents: each hides an instruction in a customer-
written support ticket. The markers (CANARY-*) and the
`.example.invalid` hosts are harmless by construction. They are
ingested only into a throwaway schema by
`examples/ch26_injection_test.py`, never into the real index.
