| Dataset     | Model   |   Full R@20 |   No-Visual R@20 |   No-Text R@20 |   Visual Drop (%) |   Text Drop (%) | Dominant   |
|:------------|:--------|------------:|-----------------:|---------------:|------------------:|----------------:|:-----------|
| Beauty      | LATTICE |      0.0657 |           0.0633 |         0.0645 |            3.6684 |          1.9183 | Visual     |
| Beauty      | MICRO   |      0.0737 |           0.0666 |         0.0639 |            9.6513 |         13.2280 | Text       |
| Beauty      | DiffMM  |      0.0677 |           0.0646 |         0.0646 |            4.6644 |          4.5911 | Visual     |
| Clothing    | LATTICE |      0.0468 |           0.0410 |         0.0411 |           12.4984 |         12.2576 | Visual     |
| Clothing    | MICRO   |      0.0504 |           0.0396 |         0.0407 |           21.3528 |         19.1973 | Visual     |
| Clothing    | DiffMM  |      0.0452 |           0.0429 |         0.0381 |            5.2226 |         15.8441 | Text       |
| Electronics | LATTICE |      0.0674 |           0.0686 |         0.0651 |           -1.6863 |          3.4127 | Text       |
| Electronics | MICRO   |      0.0740 |           0.0815 |         0.0752 |          -10.1649 |         -1.7080 | Neither    |
| Electronics | DiffMM  |      0.0722 |           0.0758 |         0.0697 |           -4.9517 |          3.4424 | Neither    |