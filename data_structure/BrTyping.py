import data_structure.Category as cat

type Composed[B:cat.Datatype=cat.Datatype, A:cat.Axis=cat.Axis] = \
    cat.Composed[cat.Array[B, A], cat.BroadcastedCategory[B, A]]
type ProductOfMorphisms[B:cat.Datatype=cat.Datatype, A:cat.Axis=cat.Axis] = \
    cat.ProductOfMorphisms[cat.Array[B, A], cat.BroadcastedCategory[B, A]]
type Rearrangement[B: cat.Datatype=cat.Datatype, A:cat.Axis=cat.Axis] = \
    cat.Rearrangement[cat.Array[B, A]]
type Block[B: cat.Datatype=cat.Datatype, A: cat.Axis=cat.Axis] = \
    cat.Block[cat.Array[B, A], cat.BroadcastedCategory[B, A]]