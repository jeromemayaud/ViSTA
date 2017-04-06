# ViSTA
A coupled vegetation/sediment-transport model for dryland environments

MODEL OVERVIEW
The coupled cellular automaton Vegetation and Sediment TrAnsport model (ViSTA) is designed to address fundamental questions about the development of arid and semi-arid landscapes in a spatially explicit way. 

The ViSTA model consists of two coupled models that interact with each other over various timescales: (i) a vegetation model that simulates vegetation growth and decline in response to environmental stresses resulting from climate and land use changes, and (ii) a sediment transport model (composed of two modules) that moves sediment across the model domain in response to spatially varying wind speeds. 

The coupled scheme is formulated such that the distribution of vegetation (Module 1) alters local wind flow characteristics (Module 2a), thus impacting sediment flux patterns over the surface (Module 2b), which in turn affects vegetation growth through ecological feedbacks. The ViSTA model also incorporates several sub-modules that can be activated to simulate herbivore/grazing, fire and precipitation events. These events have a primary impact on the state of the vegetation in Module 1.

All modules in the ViSTA model rely on local neighbourhood operations to produce dynamic responses from basic rules centred on each discrete part of the grid. All grid cells hold a variety of attributes (including sand depth, soil moisture and nutrient levels, and vegetation characteristics such as plant type, height and porosity) that are altered by applying transition rules during each timestep. The state of all cells at the end of that timestep becomes the initial state of all cells at the beginning of the next timestep. Cells interact with adjacent cells based on their local von Neumann neighbourhood.

The model is implemented in the Python® programming language. 

Fore more information regarding the theory and application of the code, see the following open-access publication: Mayaud, J. R., Bailey, R. M. & Wiggs, G. F. S. (2017). A coupled vegetation/sediment-transport model for dryland environments. Journal of Geophysical Research: Earth Surface. doi:10.1002/2016JF004096

(http://onlinelibrary.wiley.com/doi/10.1002/2016JF004096/abstract)
