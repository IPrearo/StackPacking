# StackPacking
This is a python script to generate a packing of circles within a bigger circle. This is useful for the stack-and-draw technique for making microstructured fibres.

The current code is severelly overcomplicated for what it does. If you are inclined to continue development, it would be recommended to scrap the class scructure, but keep the solving functions. This github code may be updated in the future to reflect these comments. 

## How to use
The input file is a mixture of a yaml file with a table of available diameters for the inner capillaries.

An input file will look like the example below

```
outside_diameter: 21
required: [
        {
            position_x: 0.00,
            position_y: 0.00,
            outer_diameter: 10,
            inner_diameter: 7
        },
        {
            position_x: 7.75,
            position_y: 0.00,
            outer_diameter: 5,
            inner_diameter: 3
        },
    ]
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
External Diameter	Internal Diameter
10	8
10	7
7	5
8	6
5	3
6	0
3	0
2	0
```

```outside_diameter```: defines the only required constraint, a hollow circle where all other tubes must fit.

```required```: list of obligatory tubes and their positions, each of these must have x and y positions and inner and outer diameters.

The pattern of ```=-=-```... with at least 10 characters total is used as a separator between the yaml and table styles. It should be immediately followed by the table, which has the rigid template shown above, with columns separated by tab.

The table describes the *possible* tubes that the script may choose from to place in the solution.