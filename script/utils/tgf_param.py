"""
Parameters used in the training process of the temporal graph foundation model.
"""

# supported_properties = {"edge_gs": "Edge_GS", "density_gs": "Density_GS", "node_reg": "Node_REG"}
supported_properties = {"edge_gs": "Edge_GS", "node_gs": "Node_GS", "density_gs": "Density_GS",
                        "algcnn_gs": "AlgCnn_GS", "lconcomp_gs" : "LargeConComp_GS"}

# BC -> Binary Classification
# REG -> Regression
# supported_properties_type = ["BC", "BC", "REG"]
# supported_properties_type = ["BC", "BC", "BC"]
supported_properties_type = ["BC", "BC", "BC", "BC", "BC"]

# Multi Loss combination methods
MULTI_LOSS_WEIGHTED_SUM = "weighted_sum"
MULTI_LOSS_DYNAMIC_BALANCING = "dynamic_balancing"
MULTI_LOSS_UNCERTAINTY_BASED = "uncertainty_based"
MULTI_LOSS_SIMPLE_SUM = "simple_sum"
