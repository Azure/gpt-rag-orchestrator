## Performance evaluation

Measuring performance and latency is crucial to ensure a timely and frictionless user experience. It's important to assess these factors to guarantee prompt and smooth delivery of the expected benefit.

The interactions with LLM may have several layers between the user and the LLM, like web frontend, network connections, making it vital to monitor and measure the delay at each stage. This evaluation, however, specifically focuses on the orchestrator.

## Metrics

We use the following metrics to measure performance:

- Response Time to get the answer from the user question, measured at multiple percentiles.
- Requests Per Second (RPS) for the LLM.

### Create and configure your Azure Load Testing resource

1. Create an Azure Load Testing resource.
2. Activate the Load Testing system assigned identity.
3. Add a new access policy in the solution’s Key Vault for the Load Testing identity, granting Get and List permissions for secrets.

### Configure your test suite

1. Configure `jmeter/loadtest.properties`
2. Configure `jmeter/runtest.properties`

### Run the test

run `./run-test.sh` in `jmeter` folder

## References

[How to Evaluate LLMs: A Complete Metric Framework](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/how-to-evaluate-llms-a-complete-metric-framework/)