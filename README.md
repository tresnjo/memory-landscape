# Memory Landscape
This project is an attempt to represent the physiology of the process of recalling memories in the human brain. 

The human brain consists of a complex network of neurons and synapses. Each and every event of the human experience triggers some neural path in our brains. The taste of a steak might remind you of a childhood memory, or the smell of a certain flower might remind you of the flowers your mother used to plant. Memories are recalled when the brain activates exactly those pathways associated with that memory in the brain. 

Over time, we have grasped that the brain is auto-associative. Meaning that it recalls by association. Essentially, by thinking about something associated by what we want to recall, we might be able to locate what we are seeking.

In order to simulate this recalling process, we will model this by a memory landscape. Each memory will be represented by a through on a 3D surface modelled by a gaussian distribution. Multiple of these, along with valleys, will be randomly generated over a flat surface. Furthermore, random pertubations in forms of the same valleys and throughs with smaller amplitudes will be introduced randomly. These are supposed to represent "smaller" memories in the sense that they're not as commonly found when recalling something (i.e. a dream).

To simulate the recalling process, a ball will be dropped onto the memory landscape. Using gradient descent with built-in-momentum (to simulate the fact that the "closer" you get to a memory, the faster you'll probably arrive at it, as well as being able to hop over the small pertubations which are harder to recall).

This idea came about when reading Max Tegmarks Life 3.0 and his mention of Hopfield Networks. I heavily recommend everyone to read the book.

