//! Chain multiple [`ToolRouter`]s under a single agent. Teleia uses one
//! for MCP tools and one for LSP tools, but [`teleia_agent::Agent`]
//! only accepts a single router — this collapses N routers into one
//! by name-routing through each child's `handles()` predicate.

use anyhow::{anyhow, Result};
use futures_util::future::BoxFuture;
use teleia_agent::ToolRouter;
use teleia_llm::ToolDef;

pub struct CombinedRouter {
    routers: Vec<Box<dyn ToolRouter>>,
}

impl CombinedRouter {
    pub fn new(routers: Vec<Box<dyn ToolRouter>>) -> Self {
        Self { routers }
    }
}

impl ToolRouter for CombinedRouter {
    fn definitions(&self) -> Vec<ToolDef> {
        self.routers.iter().flat_map(|r| r.definitions()).collect()
    }
    fn handles(&self, name: &str) -> bool {
        self.routers.iter().any(|r| r.handles(name))
    }
    fn dispatch<'a>(&'a mut self, name: &'a str, args: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            for r in self.routers.iter_mut() {
                if r.handles(name) {
                    return r.dispatch(name, args).await;
                }
            }
            Err(anyhow!("no router handles `{name}`"))
        })
    }
    fn set_disabled_servers(&mut self, disabled: &std::collections::BTreeSet<String>) {
        for r in self.routers.iter_mut() {
            r.set_disabled_servers(disabled);
        }
    }
    /// Ask the same router `dispatch` would pick — the first to claim the
    /// name. Asking any other one would let a permissive router vouch for
    /// a name a different router actually runs.
    fn inspects_only(&self, name: &str) -> bool {
        self.routers
            .iter()
            .find(|r| r.handles(name))
            .is_some_and(|r| r.inspects_only(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fake {
        name: &'static str,
        inspects: bool,
    }

    impl ToolRouter for Fake {
        fn definitions(&self) -> Vec<ToolDef> {
            vec![ToolDef::new(self.name, "d", json!({ "type": "object" }))]
        }
        fn handles(&self, name: &str) -> bool {
            name == self.name
        }
        fn dispatch<'a>(&'a mut self, _n: &'a str, _a: &'a str) -> BoxFuture<'a, Result<String>> {
            let who = self.name;
            Box::pin(async move { Ok(who.to_string()) })
        }
        fn inspects_only(&self, _name: &str) -> bool {
            self.inspects
        }
    }

    fn combined(first: bool, second: bool) -> CombinedRouter {
        CombinedRouter::new(vec![
            Box::new(Fake {
                name: "t",
                inspects: first,
            }),
            Box::new(Fake {
                name: "t",
                inspects: second,
            }),
        ])
    }

    #[test]
    fn inspects_only_asks_the_router_that_would_run_the_call() {
        // Two routers can claim one name; `dispatch` runs the first, so
        // the read-only claim has to come from the first too. Polling
        // until one says yes would let a permissive router vouch for a
        // call a different router actually runs — and plan mode runs an
        // inspecting call without asking anyone.
        assert!(!combined(false, true).inspects_only("t"));
        assert!(combined(true, false).inspects_only("t"));
        // A name no router claims is nobody's to vouch for.
        assert!(!combined(true, true).inspects_only("other"));
    }
}
