//! Provided automatic differentiation scaffolding (do not modify).
//!
//! Defines the [`Function`] trait, the [`Variable`] computation-graph node,
//! operator overloads (`+`, `-`, `*`, `/`, unary `-`, `.pow()`, `.log()`, `.exp()`),
//! and two complete reference examples: [`Multiply`] and [`Negate`].

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::{Add, Divide, Exp, Log, Power, Subtract, compute_gradients};

/// Base trait for automatic differentiation functions.
///
/// Implementors provide:
/// - [`Function::forward`]: evaluate the function on the scalar parent values.
/// - [`Function::backward`]: return a `Vec<f64>` containing the product of the
///   incoming derivative `grad` and each partial derivative with respect to `args`.
pub trait Function {
    /// Compute the forward pass given the parent input values.
    fn forward(&self, args: &[f64]) -> f64;

    /// Compute `[grad * df/dx_0, grad * df/dx_1, ...]` for each argument in `args`.
    ///
    /// Always returns a `Vec<f64>` with length equal to `args.len()` (even for
    /// single-argument functions).
    fn backward(&self, grad: f64, args: &[f64]) -> Vec<f64>;
}

struct VariableNode {
    value: f64,
    grad: Option<f64>,
    function: Option<Rc<dyn Function>>,
    parents: Vec<Variable>,
    num_children: usize,
}

/// A scalar node in a reverse-mode automatic differentiation computation graph.
///
/// Cloning a `Variable` is cheap (`Rc::clone`) and produces another handle to the
/// same underlying graph node.
#[derive(Clone)]
pub struct Variable {
    node: Rc<RefCell<VariableNode>>,
}

impl Variable {
    /// Create a new leaf variable with the given scalar value.
    pub fn new(value: f64) -> Self {
        Self {
            node: Rc::new(RefCell::new(VariableNode {
                value,
                grad: None,
                function: None,
                parents: Vec::new(),
                num_children: 0,
            })),
        }
    }

    /// Construct a derived node in the computation graph by applying `function`
    /// to `parents`.
    pub fn apply<F: Function + 'static>(function: F, parents: &[&Variable]) -> Self {
        let parent_values: Vec<f64> = parents.iter().map(|p| p.value()).collect();
        let value = function.forward(&parent_values);
        let parent_clones: Vec<Variable> = parents
            .iter()
            .map(|p| {
                p.node.borrow_mut().num_children += 1;
                (*p).clone()
            })
            .collect();
        Self {
            node: Rc::new(RefCell::new(VariableNode {
                value,
                grad: None,
                function: Some(Rc::new(function)),
                parents: parent_clones,
                num_children: 0,
            })),
        }
    }

    /// Scalar numerical value of this node.
    pub fn value(&self) -> f64 {
        self.node.borrow().value
    }

    /// Current accumulated gradient of this node (`None` before backprop).
    pub fn grad(&self) -> Option<f64> {
        self.node.borrow().grad
    }

    /// Set the gradient of this node.
    pub fn set_grad(&self, grad: f64) {
        self.node.borrow_mut().grad = Some(grad);
    }

    /// The [`Function`] that produced this node, or `None` for a leaf variable.
    pub fn function(&self) -> Option<Rc<dyn Function>> {
        self.node.borrow().function.clone()
    }

    /// The parent [`Variable`] handles used to compute this node (empty for leaves).
    pub fn parents(&self) -> Vec<Variable> {
        self.node.borrow().parents.clone()
    }

    /// Number of remaining child nodes whose gradients have not yet been propagated here.
    pub fn num_children(&self) -> usize {
        self.node.borrow().num_children
    }

    /// Update the remaining child count of this node.
    pub fn set_num_children(&self, num_children: usize) {
        self.node.borrow_mut().num_children = num_children;
    }

    /// Compute `self^d` using [`Power`].
    pub fn pow(&self, d: f64) -> Variable {
        Variable::apply(Power::new(d), &[self])
    }

    /// Compute the natural logarithm `ln(self)` using [`Log`].
    pub fn log(&self) -> Variable {
        Variable::apply(Log, &[self])
    }

    /// Compute the exponential `e^self` using [`Exp`].
    pub fn exp(&self) -> Variable {
        Variable::apply(Exp, &[self])
    }

    /// Run reverse-mode automatic differentiation starting from this node.
    ///
    /// Delegates to [`crate::compute_gradients`].
    pub fn compute_gradients(&self) {
        compute_gradients(self);
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.node.borrow();
        write!(f, "Variable({}, grad={:?})", node.value, node.grad)
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.node.borrow();
        f.debug_struct("Variable")
            .field("value", &node.value)
            .field("grad", &node.grad)
            .field("num_parents", &node.parents.len())
            .field("num_children", &node.num_children)
            .finish()
    }
}

// --- Operator overloads for Variable and &Variable ---

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $func:expr) => {
        impl std::ops::$trait<&Variable> for &Variable {
            type Output = Variable;
            fn $method(self, rhs: &Variable) -> Variable {
                Variable::apply($func, &[self, rhs])
            }
        }

        impl std::ops::$trait<Variable> for &Variable {
            type Output = Variable;
            fn $method(self, rhs: Variable) -> Variable {
                Variable::apply($func, &[self, &rhs])
            }
        }

        impl std::ops::$trait<&Variable> for Variable {
            type Output = Variable;
            fn $method(self, rhs: &Variable) -> Variable {
                Variable::apply($func, &[&self, rhs])
            }
        }

        impl std::ops::$trait<Variable> for Variable {
            type Output = Variable;
            fn $method(self, rhs: Variable) -> Variable {
                Variable::apply($func, &[&self, &rhs])
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Sub, sub, Subtract);
impl_binary_op!(Mul, mul, Multiply);
impl_binary_op!(Div, div, Divide);

impl std::ops::Neg for &Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Variable::apply(Negate, &[self])
    }
}

impl std::ops::Neg for Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Variable::apply(Negate, &[&self])
    }
}

// --- Provided Example Functions ---

/// Provided example (do not modify): multiplication `f(x, y) = x * y`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Multiply;

impl Function for Multiply {
    fn forward(&self, args: &[f64]) -> f64 {
        assert_eq!(args.len(), 2, "Multiply expects 2 arguments");
        let (x, y) = (args[0], args[1]);
        x * y
    }

    fn backward(&self, grad: f64, args: &[f64]) -> Vec<f64> {
        assert_eq!(args.len(), 2, "Multiply expects 2 arguments");
        let (x, y) = (args[0], args[1]);
        vec![y * grad, x * grad]
    }
}

/// Provided example (do not modify): negation `f(x) = -x`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Negate;

impl Function for Negate {
    fn forward(&self, args: &[f64]) -> f64 {
        assert_eq!(args.len(), 1, "Negate expects 1 argument");
        -args[0]
    }

    fn backward(&self, grad: f64, args: &[f64]) -> Vec<f64> {
        assert_eq!(args.len(), 1, "Negate expects 1 argument");
        vec![-grad]
    }
}
