# Agentic Personal Assistant Team

A fully local, privacy focused, team of ai agents for personal assistance.
This monorepo project aims to handle below tasks at a high level:

- Schedule events, tasks and appointments upon user request (just like calendar) while handling conflicts
- Allows grouping tasks in multiple categories: work, chores, appointments, specialdays...
- Use Eisenhower Matrix technique to prioritize tasks
- Learn from previous interactions to improve in time
- TTS and STT with client-server architecure
- Integration with other calendars: ical, outlook, etc...


## Tech Stack

Project is leveraging following apps and frameworks:

- Ollama: local model setup
- LangChain: agent orchestration
- LlamaIndex: tools, RAG and embeddings
- AI SDK: front-end
- Keycloak ?: for user management
- FastAPI: middleware
- Qdrant: vector store
- [Nettu](https://github.com/fmeringdal/nettu-scheduler): self-hosted calndar and scheduler server with rest api

## Running the App

### Using Docker

TODO: update steps and ref to docker-compose.yml

### Manual Setup

TODO: ref to documentation of individual apps like ollama, qdrant then explain how to confiure


## Setting up Dev Environment

TODO: links to all readmes