//! Building a tree-level [`Diagram`] from its parts.
//!
//! A diagram normally comes out of [`DiagramGenerator`](crate::diagram::DiagramGenerator). A program that keeps
//! diagrams in its own representation (stores them, relabels their legs, glues decay chains onto them) can rebuild
//! a [`Diagram`] here and hand its [`DiagramView`] to the drawing and inspection methods.
//!
//! The parts follow the generator's conventions:
//!
//! - every vertex lists its attached lines in [`Vertex::propagators`]: a propagator by its index, an external leg
//!   `i` (incoming legs first, then outgoing) as `i - n_ext`, which is negative;
//! - a [`Leg`] names the internal vertex it attaches to, its particle (the antiparticle of the outgoing state for an
//!   outgoing leg, since every leg is read incoming) and its momentum as a unit vector over the external momenta;
//! - a [`Propagator`] names its two vertices, the particle flowing from `vertices[0]` to `vertices[1]`, and its
//!   momentum as a combination of the external momenta with the outgoing components' signs flipped.
//!
//! Only tree diagrams are accepted. At tree level the derived fields are exact without a topology search: every
//! propagator is a bridge, and both symmetry factors are one, since a tree whose external legs are labelled has no
//! nontrivial automorphism and no two propagators joining the same pair of vertices.

use std::fmt;

use crate::diagram::view::DiagramView;
use crate::diagram::{Diagram, Leg, Propagator, Vertex};
use crate::model::Model;

/// Why a set of parts does not describe a tree diagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagramPartsError {
    /// The diagram has no internal vertex. The single-propagator diagram has no parts to build it from.
    NoVertices,
    /// A leg or propagator names a particle the model does not have.
    UnknownParticle { line: String, particle: usize },
    /// A vertex names an interaction the model does not have.
    UnknownInteraction { vertex: usize, interaction: usize },
    /// A leg or propagator names a vertex index past the vertex list.
    UnknownVertex { line: String, vertex: usize },
    /// A momentum vector does not have one entry per external leg.
    MomentumLength { line: String, len: usize, n_ext: usize },
    /// A vertex lists a line index that is neither an external leg nor a propagator.
    UnknownLine { vertex: usize, line: isize },
    /// A vertex lists a line whose own endpoints do not include that vertex.
    NotAttached { vertex: usize, line: String },
    /// A line is listed by the wrong number of vertices: once for a leg, twice for a propagator.
    Incidence {
        line: String,
        listed: usize,
        expected: usize,
    },
    /// A propagator starts and ends on the same vertex.
    SelfLoop { propagator: usize },
    /// The parts are not a tree: the propagator count is not one less than the vertex count, or the vertices are
    /// not connected.
    NotATree { vertices: usize, propagators: usize },
}

impl fmt::Display for DiagramPartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVertices => write!(f, "a diagram built from parts needs at least one vertex"),
            Self::UnknownParticle { line, particle } => {
                write!(f, "{line} names particle {particle}, which the model does not have")
            }
            Self::UnknownInteraction { vertex, interaction } => {
                write!(
                    f,
                    "vertex {vertex} names interaction {interaction}, which the model does not have"
                )
            }
            Self::UnknownVertex { line, vertex } => write!(f, "{line} names vertex {vertex}, which does not exist"),
            Self::MomentumLength { line, len, n_ext } => {
                write!(f, "{line} has a momentum of {len} entries for {n_ext} external legs")
            }
            Self::UnknownLine { vertex, line } => {
                write!(
                    f,
                    "vertex {vertex} lists line {line}, which is neither a leg nor a propagator"
                )
            }
            Self::NotAttached { vertex, line } => {
                write!(f, "vertex {vertex} lists {line}, which is not attached to it")
            }
            Self::Incidence { line, listed, expected } => {
                write!(f, "{line} is listed by {listed} vertices instead of {expected}")
            }
            Self::SelfLoop { propagator } => write!(f, "propagator {propagator} starts and ends on one vertex"),
            Self::NotATree { vertices, propagators } => write!(
                f,
                "{vertices} vertices and {propagators} propagators do not form a connected tree"
            ),
        }
    }
}

impl std::error::Error for DiagramPartsError {}

impl Diagram {
    /// Build a tree-level diagram from its legs, propagators and vertices, in the conventions described in the
    /// [module documentation](self). The sign is computed as the generator computes it; every propagator is a
    /// bridge and both symmetry factors are one.
    ///
    /// The parts are checked for consistency (indices in range, every line attached where the vertices say, a
    /// connected tree), not against the model's interactions: a vertex whose particles do not match its
    /// interaction is accepted, and draws as given.
    pub fn from_tree_parts(
        model: &Model,
        incoming_legs: Vec<Leg>,
        outgoing_legs: Vec<Leg>,
        propagators: Vec<Propagator>,
        vertices: Vec<Vertex>,
    ) -> Result<Self, DiagramPartsError> {
        check_tree_parts(model, &incoming_legs, &outgoing_legs, &propagators, &vertices)?;
        let bridges = (0..propagators.len()).collect();
        let mut diagram = Diagram {
            incoming_legs,
            outgoing_legs,
            propagators,
            vertices,
            vertex_symmetry: 1,
            propagator_symmetry: 1,
            bridges,
            sign: 1,
        };
        diagram.sign = DiagramView::new(model, &diagram, &Vec::new()).calculate_sign();
        Ok(diagram)
    }

    /// A view of this diagram against `model`, the model its particle and interaction indices refer to.
    /// `momentum_labels` names the external momenta, one per leg, for the view's momentum strings; the drawing
    /// methods do not read them.
    pub fn view<'a>(&'a self, model: &'a Model, momentum_labels: &'a Vec<String>) -> DiagramView<'a> {
        DiagramView::new(model, self, momentum_labels)
    }
}

fn check_tree_parts(
    model: &Model,
    incoming_legs: &[Leg],
    outgoing_legs: &[Leg],
    propagators: &[Propagator],
    vertices: &[Vertex],
) -> Result<(), DiagramPartsError> {
    if vertices.is_empty() {
        return Err(DiagramPartsError::NoVertices);
    }
    let n_particles = model.particles_iter().count();
    let n_ext = incoming_legs.len() + outgoing_legs.len();
    let legs: Vec<&Leg> = incoming_legs.iter().chain(outgoing_legs).collect();

    for (i, leg) in legs.iter().enumerate() {
        let line = format!("leg {i}");
        if leg.particle >= n_particles {
            return Err(DiagramPartsError::UnknownParticle {
                line,
                particle: leg.particle,
            });
        }
        if leg.vertex >= vertices.len() {
            return Err(DiagramPartsError::UnknownVertex {
                line,
                vertex: leg.vertex,
            });
        }
        if leg.momentum.len() != n_ext {
            return Err(DiagramPartsError::MomentumLength {
                line,
                len: leg.momentum.len(),
                n_ext,
            });
        }
    }
    for (i, prop) in propagators.iter().enumerate() {
        let line = format!("propagator {i}");
        if prop.particle >= n_particles {
            return Err(DiagramPartsError::UnknownParticle {
                line,
                particle: prop.particle,
            });
        }
        if let Some(&v) = prop.vertices.iter().find(|&&v| v >= vertices.len()) {
            return Err(DiagramPartsError::UnknownVertex { line, vertex: v });
        }
        if prop.momentum.len() != n_ext {
            return Err(DiagramPartsError::MomentumLength {
                line,
                len: prop.momentum.len(),
                n_ext,
            });
        }
        if prop.vertices[0] == prop.vertices[1] {
            return Err(DiagramPartsError::SelfLoop { propagator: i });
        }
    }

    let mut leg_listed = vec![0usize; n_ext];
    let mut prop_listed = vec![0usize; propagators.len()];
    for (v, vertex) in vertices.iter().enumerate() {
        if vertex.interaction >= model.n_vertices() {
            return Err(DiagramPartsError::UnknownInteraction {
                vertex: v,
                interaction: vertex.interaction,
            });
        }
        for &line in &vertex.propagators {
            if line < 0 {
                let leg = line + n_ext as isize;
                if leg < 0 {
                    return Err(DiagramPartsError::UnknownLine { vertex: v, line });
                }
                let leg = leg as usize;
                if legs[leg].vertex != v {
                    return Err(DiagramPartsError::NotAttached {
                        vertex: v,
                        line: format!("leg {leg}"),
                    });
                }
                leg_listed[leg] += 1;
            } else {
                let prop = line as usize;
                if prop >= propagators.len() {
                    return Err(DiagramPartsError::UnknownLine { vertex: v, line });
                }
                if !propagators[prop].vertices.contains(&v) {
                    return Err(DiagramPartsError::NotAttached {
                        vertex: v,
                        line: format!("propagator {prop}"),
                    });
                }
                prop_listed[prop] += 1;
            }
        }
    }
    for (i, &listed) in leg_listed.iter().enumerate() {
        if listed != 1 {
            return Err(DiagramPartsError::Incidence {
                line: format!("leg {i}"),
                listed,
                expected: 1,
            });
        }
    }
    for (i, &listed) in prop_listed.iter().enumerate() {
        if listed != 2 {
            return Err(DiagramPartsError::Incidence {
                line: format!("propagator {i}"),
                listed,
                expected: 2,
            });
        }
    }

    let not_a_tree = DiagramPartsError::NotATree {
        vertices: vertices.len(),
        propagators: propagators.len(),
    };
    if propagators.len() + 1 != vertices.len() {
        return Err(not_a_tree);
    }
    let mut reached = vec![false; vertices.len()];
    let mut stack = vec![0usize];
    reached[0] = true;
    while let Some(v) = stack.pop() {
        for prop in propagators.iter().filter(|p| p.vertices.contains(&v)) {
            let w = if prop.vertices[0] == v {
                prop.vertices[1]
            } else {
                prop.vertices[0]
            };
            if !reached[w] {
                reached[w] = true;
                stack.push(w);
            }
        }
    }
    if reached.iter().all(|&r| r) {
        Ok(())
    } else {
        Err(not_a_tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagram::DiagramGenerator;

    fn rebuild(model: &Model, d: &Diagram) -> Result<Diagram, DiagramPartsError> {
        Diagram::from_tree_parts(
            model,
            d.incoming_legs.clone(),
            d.outgoing_legs.clone(),
            d.propagators.clone(),
            d.vertices.clone(),
        )
    }

    /// Every generated tree diagram rebuilds to the same diagram, sign and symmetry factor included, and draws to
    /// the same SVG. The processes cover fermion lines with crossed legs (sign), four-point vertices and
    /// several propagators.
    #[test]
    fn generated_tree_diagrams_round_trip() {
        let processes: [(&[&str], &[&str]); 4] = [
            (&["u", "u~"], &["e-", "e+", "g"]),
            (&["g", "g"], &["g", "g", "g"]),
            (&["u", "d"], &["u", "d", "g"]),
            (&["e+", "e-"], &["W+", "W-", "Z"]),
        ];
        let mut checked = 0;
        for (incoming, outgoing) in processes {
            let generator = DiagramGenerator::new(incoming, outgoing, 0, Model::default(), None).unwrap();
            let container = generator.generate();
            assert!(container.len() > 0, "{incoming:?} > {outgoing:?} generated nothing");
            let model = container.model.as_ref().unwrap();
            for view in container.views() {
                let rebuilt = rebuild(model, view.diagram).unwrap();
                let mut expected = view.diagram.clone();
                expected.bridges.sort_unstable();
                assert_eq!(rebuilt, expected, "{incoming:?} > {outgoing:?}");
                let labels = Vec::new();
                assert_eq!(rebuilt.view(model, &labels).draw_svg_string(), view.draw_svg_string());
                checked += 1;
            }
        }
        assert!(checked > 50, "only {checked} diagrams checked");
    }

    /// The sign is recomputed from the parts, not carried over: swapping two identical outgoing fermions' legs
    /// flips it.
    #[test]
    fn sign_follows_the_parts() {
        let generator = DiagramGenerator::new(&["u", "u"], &["u", "u"], 0, Model::default(), None).unwrap();
        let container = generator.generate();
        let model = container.model.as_ref().unwrap();
        let d = container.views().next().unwrap().diagram.clone();
        let mut swapped = d.clone();
        swapped.outgoing_legs.swap(0, 1);
        let n_ext = 4isize;
        for vertex in &mut swapped.vertices {
            for line in &mut vertex.propagators {
                if *line == 2 - n_ext {
                    *line = 3 - n_ext;
                } else if *line == 3 - n_ext {
                    *line = 2 - n_ext;
                }
            }
        }
        let original = rebuild(model, &d).unwrap();
        let rebuilt = rebuild(model, &swapped).unwrap();
        assert_eq!(rebuilt.sign(), -original.sign());
    }

    #[test]
    fn inconsistent_parts_are_refused() {
        let generator = DiagramGenerator::new(&["u", "u~"], &["e-", "e+"], 0, Model::default(), None).unwrap();
        let container = generator.generate();
        let model = container.model.as_ref().unwrap();
        let d = container.views().next().unwrap().diagram.clone();

        let mut detached = d.clone();
        detached.incoming_legs[0].vertex = 1 - detached.incoming_legs[0].vertex;
        assert!(matches!(
            rebuild(model, &detached),
            Err(DiagramPartsError::NotAttached { .. })
        ));

        let mut short = d.clone();
        short.propagators[0].momentum.pop();
        assert!(matches!(
            rebuild(model, &short),
            Err(DiagramPartsError::MomentumLength { .. })
        ));

        let mut unlisted = d.clone();
        unlisted.vertices[0].propagators.retain(|&l| l >= 0);
        assert!(matches!(
            rebuild(model, &unlisted),
            Err(DiagramPartsError::Incidence { .. })
        ));

        let mut looped = d.clone();
        looped.propagators[0].vertices = [0, 0];
        assert!(matches!(
            rebuild(model, &looped),
            Err(DiagramPartsError::SelfLoop { .. })
        ));

        assert_eq!(
            Diagram::from_tree_parts(model, vec![], vec![], vec![], vec![]),
            Err(DiagramPartsError::NoVertices)
        );
    }
}
