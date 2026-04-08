# pyncd

```python
import construction_helpers as ch # Needed for algebraic manipulation
import data_structure.Category as cat
import data_structure.Operators as ops


qk_matmul = ops.Einops.template('q h d, x h d -> h q d')
softmax = ops.SoftMax.template()
mask = ops.WeightedTriangularLower()
sv_matmul = ops.Einops.template('h q x, x h d -> q h d')
attention_core = qk_matmul @ softmax @ mask @ sv_matmul
```
 ![The Pythonic algebraic expression above generates an aligned attention algorithm.](_guide/figures/alignment.png)

## Description

This is a package for formally expressing deep learning models based on [Neural Circuit Diagrams](https://openreview.net/forum?id=RyZB4qXEgt), [FlashAttention on a Napkin](https://openreview.net/forum?id=pF2ukh7HxA), [Spherical Attention](https://arxiv.org/abs/2505.09326) and a [GPU Mode presentation](https://www.youtube.com/watch?v=hAoY2bpRIKg). The main goal of this package is to provide a simple and intuitive way to define and visualize deep learning models, while also allowing for formal reasoning about their properties. In `data_structure`, you will find a high-level implementation of the structural aspects of deep learning models.

The other folders provide utilities. These are;

 - `construction_helpers`: Allows models to be defined via operator overloading, `@` (for sequential composition), `*` (for parallel "products") and `>>` (for batch lifting). When using `@`, axes are automatically aligned.
 - `data_transfer` and `websocket_transfer`: These packages provide JSON encoding and communication over WebSockets, integrating with the [`tsncd`](https://github.com/mit-zardini-lab/tsncd) package for displaying diagrams.
 - `torch_compile`: This package allows algebraic descriptions to be converted into PyTorch modules.
 - `display`: This package allows for textual display of algebraic expressions.
 - `graphs`: This package implements the mathematical process of [converting morphisms in a symmetric monoidal category into hypergraphs](https://arxiv.org/pdf/2305.08768), which opens up flexible algebraic manipulation in the future.

These utilities build on the core data structure. They feed into a "web" of tools that allow for algebraic manipulation, diagrammatic visualization, and execution of deep learning models. For instance, we can compose from algebraic constructs to the Python data structure, to PyTorch or diagrammatic visualizations. Given the underlying mathematical structure of the data structure, we can imitate mathematical transforms such as product categories to hypergraphs.

*The modularized tools in this package generate a "web" of features which integrate into each other.*
![alt text](_guide/figures/the_web_tsncd.png)

 ## The Structure
 We implement mathematical expressions with `Term`s. We keep everything in a high-level structure, and leave evaluation to specific tools such as the Torch compiler or TypeScript diagramming. Deep learning models and their components are morphisms in the `BroadcastedCategory`. (Product) Categories are compositional structures that allow for components called **morphisms** to be sequentially **composed** and placed into parallel **products**, forming new morphisms. Composition is anchored by **objects**. We also have a special morphism to **rearrange** objects in a product.
 
 The structure of a deep learning model consists of these constructed terms ultimately referencing seed morphisms. In the case of the `BroadcastedCategory`, representing deep learning models, the seed morphisms are single, broadcasted operations called `Broadcasted`, and the objects are `Array`s.

 The broadcasting semantics of this package are defined through `Weave`s and the `StrideCategory`. Weaves indicate which axes are tiled or form part of the "target" operation. The output along indexes of tiled axes are defined relative to indexes along the inputs by passing through a morphism of the `StrideCategory`, which corresponds to an affine transform. This relationship may be direct, for example, the `i, j` output index may correspond to the `i, j` input index, or they may take a stride manipulation. These are always affine. This allows more complex patterns to be enforced. The `i, j` output index may correspond to the `j, i` input index, enforcing a transpose. The `i` output index may correspond to the `i, i` input index, giving a diagonalization. Or, the `x, k` output index may correspond to the `x + k` input index, giving a convolution.
 
  *Here, the p0, p1 indexes of the output correspond to the reindexed locations along the input.*
 ![The p0, p1 indexes of the output correspond to the reindexed locations along the input.]( _guide/figures/broadcast_weave.png)

# Setup
 - ***This package requires Python 3.14.***

 - For diagrammatic visualization, the [`tsncd`](https://github.com/mit-zardini-lab/tsncd) package is required. We run `python run_server.py` (requires `pip install websockets`) in an independent terminal. When the server is running, the browser connects on refresh. As in `Transformer.ipynb`, we connect from Python, updating the data in the server.

 - The `minimum_working_example.py` both hosts a server and has command line inputs for various components which are sent to the server for visaulization.

 - For compiling deep learning models, [PyTorch](https://pytorch.org/) and [Einops](https://einops.rocks/) (`pip install einops`) are required.