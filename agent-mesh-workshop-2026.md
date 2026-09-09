---
layout: default
title: "Agent Mesh Workshop 2026 | Flurin Gishamer"
description: "Materials and exercises for the Agent Mesh workshop on 10 September 2026."
permalink: /agent-mesh-workshop-2026/
---

# Agent Mesh Workshop 2026

Welcome to the page for Workshop Tage 2026: the Agent Mesh course on 10 September 2026.

We will move from the parts of an agent system to the control plane that lets many agents work safely together. Expect a practical day: we will inspect capabilities, work through real integration boundaries, introduce a gateway, and use evidence to understand what happened in a run.

## Course materials

Download the [slides and participant handout](/downloads/agent-mesh-workshop-2026-materials.zip). The archive is password protected; you will receive the password during the course. If you do not have it, [email me](mailto:flurin@gishamer.io?subject=Agent%20Mesh%20Workshop%202026%20archive%20password) and I will send it to you.

## Setup instructions

To run the exercises on your own laptop, clone the [Agent Mesh setup repository](https://github.com/gishamer/agent-mesh-setup#readme) and then follow its README. Before you begin, install the prerequisites that the README does not explain how to install:

- [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/), [Visual Studio Code](https://code.visualstudio.com/docs/setup/mac), and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- The required command-line tools: [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/), [kind](https://kind.sigs.k8s.io/docs/user/quick-start/), [Helm](https://helm.sh/docs/intro/install/), [Node.js and npm](https://nodejs.org/en/download), [jq](https://jqlang.org/download/), [Python 3](https://www.python.org/downloads/macos/), and [Git](https://git-scm.com/downloads/mac).
- macOS already includes `curl`. If `make --version` does not work, install Apple's Command Line Tools with `xcode-select --install`.
