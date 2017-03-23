"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


This module provides a dictionary to store data that is cumbersome
to be passed around through function calls but at the same time should
only be computed once. The dictionary could hold the data for as long
as it is required.

1-12-2017
"""

Globals = {}
