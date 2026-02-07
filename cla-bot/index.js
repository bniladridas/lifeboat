const core = require("@actions/core");
const github = require("@actions/github");
const fs = require("fs");

async function run() {
  try {
    const token = core.getInput("github-token");
    const octokit = github.getOctokit(token);
    const { owner, repo } = github.context.repo;
    const pr = github.context.payload.pull_request;

    if (!pr) {
      core.info("Not a PR event, skipping");
      return;
    }

    const author = pr.user.login;
    const signers = JSON.parse(
      fs.readFileSync("cla-signers.json", "utf8")
    ).signed;

    if (signers.includes(author)) {
      core.info(`CLA signed by ${author}`);
      return;
    }

    await octokit.rest.issues.createComment({
      owner,
      repo,
      issue_number: pr.number,
      body: `❌ **CLA required**

Hi @${author}, please sign the Contributor License Agreement to proceed.

👉 https://example.com/cla

Once signed, this check will pass automatically.`
    });

    core.setFailed(`CLA not signed by ${author}`);
  } catch (err) {
    core.setFailed(err.message);
  }
}

run();
