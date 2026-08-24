# agent

Python. FastAPI + LangChain. Turns a sentence into calls against connected services.

## Responsibility

Receive a message plus the list of services this user has authorized, decide which to
reach for, execute, and answer. There is no service picker in the UI — choosing is
this service's job.

## Design

- **One endpoint, every provider.** `POST /chat`. Providers are not separate routes;
  they are tools registered on a single agent. Adding a provider means registering
  tools, not adding an endpoint.
- **Tools are generated per request** from the user's connected services. A user with
  only Gmail connected gets an agent that cannot see YouTube tools at all — the
  narrowest capability set that answers the question.
- **Credentials never arrive here.** Tools call `services/credentials` to perform
  authorized requests. If this service is compromised, no token leaks with it.

## Trust

This is the least trusted component in the system: it executes model-directed control
flow over user data. It gets the smallest surface deliberately — no master key, no
direct database writes to credential tables, no ability to send on the user's behalf.

## Scope limit

Reads, searches, and drafts. A draft is returned for the user to approve. The agent
does not send messages, post, or mutate state on a connected service.
