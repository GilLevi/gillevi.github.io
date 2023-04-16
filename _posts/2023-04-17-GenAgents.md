---
title: 'Generative Agents: Interactive Simulacra of Human Behavior'
date: 2023-04-17
permalink: /posts/GenAgents
tags:
  - ChatGPT
  - Research
  - Papers
  - LLM
---

This paper is one of the most exciting and creative works I’ve read recently. In a nutshell, it shows that we can leverage Large Language Models (LLM) and specifically ChatGPT [] to create agents displaying human-like behaviour. Each agent has different persona, occupation, hobbies, family and friends and the agents “live’ a small neighbourhood (implemented as a Sims-like environment)  including houses, a supply store, a bar, a college, etc.  As the agents go about their day (wake up, brush teeth, eat breakfast, go to work, buy groceries, etc.) they interact with each other in conversation, create new memories, form opinions and reflect on past memories as the plan ahead. Not only the agents display human-like behaviour, emerging social behaviours emerge, for example: when one agent plans a Valentine party, the information propagate, agents invite each other to the party and coordinate when to arrive (one agent even invites her crush to the party and the couple goes on a date!). The authors achieve that by implementing an architecture which includes <b>a long-term memory component</b> (in natural language) allowing to store and retrieve relevant memories, a <b>reflection component</b> that synthesise the memory to create conclusions (todo: better explain) and a *planning component* which uses those reflections to creates high-level actions and reactions based, which are then further detail and to the memory component. The implementation does not include any Reinforcement Learning component (apart from the RLHF[] training in ChatGPT), which is very intriguing showing that a believable and even goal-oriented behaviour (e.g. throwing a party) can be modelled using LLM alone.


### Further description of the agents and the environment 
Below I’ll further explain the agents and the environment setting and “implementation details”. If you are familiar with the game “The-Sims” (or similar), you can jump ahead to the next section as the game environment is extremely similar and very intuitive.

#### Agent "Initialisation”
As described above, the agents are represented by avatars inhabiting a small town named SmallVille. 25 avatars inhabit the town, each one described by a unique one-paragraph natural language description depicting the agent’s identify, occupation and relationship with other agents as seed memories. For example, below is the description for the agent named “John Lin”: 

John Lin is a pharmacy shopkeeper at the Willow Market and Pharmacy who loves to help people. He is always looking for ways to make the process of getting medication easier for his customers; John Lin is living with his wife, Mei Lin, who is a college professor, and son, Eddy Lin, who is a student studying music theory; John Lin loves his family very much; John Lin has known the old couple next-door, Sam Moore and Jennifer Moore, for a few years; John Lin thinks Sam Moore is a kind and nice man; John Lin knows his neighbor, Yuriko Yamamoto, well; John Lin knows of his neighbors, Tamara Taylor and Carmen Ortiz, but has not met them before; John Lin and Tom Moreno are colleagues at The Willows Market and Pharmacy; John Lin and Tom Moreno are friends and like to discuss local politics together; John Lin knows the Moreno family somewhat well — the husband Tom Moreno and the wife Jane Moreno.

#### Agents movement and interaction
At each time step, the agents output an action describe and natural language which is translated to an icon displayed next to the character’s avatar (e.g. Isabella Rodriguez is writing in her journal" which displayed as XX). Whenever agents are in the same local area, they are aware of each other and decide whether to engage in conversion. Agents move freely in SmallVile where the generative architectures determines when and where an agent should move to and the sandbox game engine will compute a walking path to the desired location and the movement will be presented on the screen. Agents also influence the objects in the world, for example a bed can be occupied when an agent is sleeping. 


#### User input and control
A user (human user that is…) can provide input and interact in the simulation by communicating with an agent as a specified persona (e.g. “news reporter”) or directing the simulating by issuing an “inner voice” command to an agent, for example the agent “John” is issued the “inner-voice” command “You are going to run against Sam in the upcoming election” and following decides to run in the election and shares the decision with the agents playing the role of his wife and son. Users also have the ability to change the state of an object through a natural language command in a pre-determined syntax and can enter the sandbox game as avatars. 





