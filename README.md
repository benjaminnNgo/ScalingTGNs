# UTG
Unifying Temporal Graph (UTG) comparison between Continuous Time Dynamic Graphs and Discrete Time Dynamic Graphs 

### CTDG Evaluation Setting

1. convert CTDG edgelist into DTDG edgelist (for the training set)

2. store the converted DTDG edgelist (with converted UNIX timestamps)

3. load the DTDG training set with TGB framework (or construct separate data loading / data class)

4. train TGN on DTDG training set

5. use the TGB class for evaluation set (the test edges) and evaluation
