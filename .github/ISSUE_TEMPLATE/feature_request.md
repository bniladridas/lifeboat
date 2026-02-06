---
name: Bug Report
description: Report a bug in the Lifeboat project.
labels: bug
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: What happened?
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to Reproduce
      description: How did you encounter the bug?
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What should have happened?
  - type: textarea
    id: actual
    attributes:
      label: Actual Behavior
      description: What actually happened?
  - type: input
    id: version
    attributes:
      label: Version
      description: Which version were you using?
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: |
        This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org). By participating, you are expected to uphold this code.
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
---
name: Feature Request
description: Suggest a new feature for Lifeboat.
labels: enhancement
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: What feature would you like to see added?
    validations:
      required: true
  - type: textarea
    id: rationale
    attributes:
      label: Rationale
      description: Why is this feature useful? What problem does it solve?
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives
      description: Have you considered any alternative solutions?
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: |
        This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org). By participating, you are expected to uphold this code.
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
