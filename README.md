### README

This repository contains Python code for solving a vehicle routing and energy management problem using Pyomo. The problem involves optimizing the allocation of shared autonomous electric vehicles (SAEVs) for passenger transportation while managing their energy resources efficiently. The code utilizes mathematical programming techniques to formulate and solve the optimization model.

#### Problem Description
The problem addresses the allocation of SAEVs for passenger transportation in an urban area. It involves optimizing various aspects such as vehicle routing, charging/discharging strategies, and energy trading with the grid during peak demand periods or outages.

#### Code Structure
- `main.py`: Contains the main code for formulating and solving the optimization model using Pyomo.
- `functions.py`: Defines helper functions for generating random coordinates, creating distance matrices, and plotting nodes on a map.
- `requirements.txt`: Lists the required Python packages for running the code.
- `README.md`: This file provides an overview of the repository and instructions for usage.

#### Dependencies
- Pyomo: A Python-based open-source optimization modelling language.
- Folium: A Python library for creating interactive maps.
- Matplotlib: A Python plotting library for creating static, animated, and interactive visualizations.

#### Usage
To run the code, ensure that Python and the required dependencies are installed on your system. You can install the dependencies using the following command:
```
pip install -r requirements.txt
```
Once the dependencies are installed, you can execute the `main.py` script to formulate and solve the optimization model.

#### Acknowledgments
This code was developed as part of a research project aimed at addressing urban transportation and energy management challenges. We acknowledge the support and contributions of the research team involved in this project.

#### License
This code is provided under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use and modify it for your own projects.

For any questions or issues, please contact [the repository owner](https://github.com/jmanzolli).
