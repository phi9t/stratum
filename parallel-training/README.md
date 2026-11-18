# Parallel training explorer

The page combines a Qwen3 stage scheduling scenario with an independent graph inspector.
Run npm ci, npm test and npm run build to build local JavaScript and CSS into dist/.
Run the behavioral browser check from the repository root with
uv run --frozen python parallel-training/tests/browser.py after building.

## Semantic graph and projections

src/graph/semantic.js defines stratum-semantic-graph/v1. Model operations have stable
module and typed port identities. Each dependency names its exact source port,
destination port, tensor and role. Tensor records hold numeric axes, precision and
layout, or an explicit reason for an unknown shape. Parameters have their own identity.

The realization holds the declared mesh, explicit rank memberships, module placement,
communication operations and policy. Execution instances add step, microbatch, phase
and rank to operation identity. Observations are a separate collection, empty in these
fixtures. Scenario invocation ids are scoped to the graph, layer, step, microbatch and
rank. They establish no kernel identity or cross-run correlation.

src/graph/projection.js contracts modules for display. It retains every external edge,
its original endpoints and its boundary ports. Multiple tensors between the same visible
modules stay independent. Internal dependencies remain in the contraction result.
Selection stores the canonical operation and survives collapsing its module.
src/graph/layout.js uses ELK, the layout engine used by the unlanded plan explorer.
Only the layout result contains coordinates and routed geometry.

src/graph/GraphExplorer.jsx is the common XYFlow renderer and tensor/dependency inspector.
It provides module collapse, dependency selection and an evidence/execution panel.
src/graph/state.js validates URL state. Back/Forward restores graph, selected operation,
collapsed modules and panel. Model definitions and projection geometry carry no UI state.

## Fixtures and support

- **Qwen3 MLP** preserves the published unfused logical forward/backward scenario and the
  gate/up/down saved-tensor branches. Shapes are supplied as N, D and TP-local K.
  Layer-scoped FSDP collectives include parameters outside the MLP.
  Gathered-weight retention changes the availability dependency. TP and FSDP memberships
  are explicit, selectable rank groups. Forward topology and TP convention link to
  pinned reference code. Those references carry no source acceptance or runtime approval.
- **Attention + residual** is an illustrative single-head causal scaled-dot-product
  forward graph with authored dimensions, projection branches and a residual edge.
  The attention paper is a mathematical reference. This fixture claims no checkpoint
  architecture or numerical parity.
- **Sparse lookup + dense mixer** is an illustrative Civet-style teaching motif with
  integer lookup indices, a table, a dense sigmoid gate and a residual.
  It has no model implementation source or distributed placement. It claims no Civet
  architecture, backward, runtime or parity coverage.

The alternative fixtures change the dependency inspector. The schedule, activation
inventory and stage handoffs continue to describe the explicit Qwen3 scenario above.
The schedule's retained-state counts do not establish activation bytes or operator timing.

## Existing plan engine integration

src/graph/plan-adapter.js accepts the existing buildPlanGraph() result from the
unlanded tools/parallel-plan.mjs without recomputing its parameter or traffic ledger.
It maps the existing namespace hierarchy to modules and preserves upstream output ids.
The old exporter reuses input id 0 for multiple incoming edges. The adapter gives each
logical input occurrence its own port and retains the original upstream identifiers.
It preserves absent output shapes as unknown. It supplies forward plan structure only.

This adapter is the integration boundary for the existing Model Explorer/ELK stack.
It does not land that draft exporter or replace its arithmetic. The published page's
fixtures and the adapter use the same semantic validation and contraction/layout APIs.
The old endpoint-pair edge deduplication is unsuitable for tensor dependency identity.

Tests cover invalid schemas/endpoints/shapes, MLP branches, dimension changes, fixture
scope, rank groups, contraction, deterministic edge identity, ELK node geometry/routes,
and browser navigation, keyboard-accessible controls and mobile overflow.
No automatic FX/autograd capture, stream scheduling or runtime import is provided.
