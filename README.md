# 3W Well Agent

This repository contains the code for the 3W Well Agent. It uses the [3W dataset from Vargas et al. 2019](https://www.sciencedirect.com/science/article/pii/S0920410519306357). The dataset is available on [Petrobras's github page](https://github.com/petrobras/3W). The goal is to demonstrate not only how to train and deploy machine learning models on Databricks, but also how to incorporate them (and the underlying data)into an agentic framework. 

## Contents

### 1. Download
This section downloads the 3W dataset from Petrobras's github page

### 2. Process
This job processes the dataset into a single table that we can use for machine learning.

### 3. Train
This job trains the hydrate prediction model for each well. We split the dataset into three parts: training, validation, and test. We use the validation set to tune the model and the test set to evaluate the model via the agent framework and user testing (since we don't have new data being generated).

### 4. Deploy
This job deploys the machine learning models.

### 5. Agentify
This sets up an agent to use the deployed models and datasets via Unity Catalog and LangGraph so we can chat with our data and predictions.

## Getting started

0. Install UV: https://docs.astral.sh/uv/getting-started/installation/

1. Install the Databricks CLI from https://docs.databricks.com/dev-tools/cli/databricks-cli.html

2. Authenticate to your Databricks workspace, if you have not done so already:
    ```
    $ databricks configure
    ```

3. To deploy a development copy of this project, type:
    ```
    $ databricks bundle deploy --target dev
    ```
    (Note that "dev" is the default target, so the `--target` parameter
    is optional here.)

    This deploys everything that's defined for this project.
    For example, the default template would deploy a job called
    `[dev yourname] hydrate_job` to your workspace.
    You can find that job by opening your workpace and clicking on **Workflows**.

4. Similarly, to deploy a production copy, type:
   ```
   $ databricks bundle deploy --target prod
   ```

   Note that the default job from the template has a schedule that runs every day
   (defined in resources/hydrate.job.yml). The schedule
   is paused when deploying in development mode (see
   https://docs.databricks.com/dev-tools/bundles/deployment-modes.html).

5. To run a job or pipeline, use the "run" command:
   ```
   $ databricks bundle run
   ```
6. Optionally, install developer tools such as the Databricks extension for Visual Studio Code from
   https://docs.databricks.com/dev-tools/vscode-ext.html.

7. For documentation on the Databricks asset bundles format used
   for this project, and for CI/CD configuration, see
   https://docs.databricks.com/dev-tools/bundles/index.html.
