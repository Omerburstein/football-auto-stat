1. [x] Create the following skills, and create an hook for them to make sure you use them when new code is added: (done 2026-10-27)
    i. [x] create tests - adds unit, integration, and algo tests
    ii. [x] clean up code - keeps files short, removes unused imports, keeps oop techniques, removing redundant code, having informative comments. Can recive a file or folder to work on specifically
    iii. [x] create overlay video - running the algo, showing the final video containing all the relevant stats and the bbox of the players, ball and goals
    iv. [x] Deal with todo, writing new things to the file, maintaining order, marking as completed
2. [x] Create the following agents, I want them to use by default Opus 5.5 on high mode: (done 2026-10-27)
    i. [x] detection agent - will work on detecting and tracking the players, ball goals
    ii. [x] stats agent - will work on detecting the existance of the conditions in the stats (goals, shots, etc)
    iii. [x] integration agent - will later work on a way to display the results, either on the computer or by making a apk file for phones
3. [x] I want to create a way of detecting and tracking soccer players by one or two cellphone cameras (for now build it for only one, but make the design versatile for changes). You can use the following assumptions: (done 2026-10-27)
    i. The max amount of players on the field will be given to you
    ii. Each player is having the same outfit (design it so it will be also able to change in the middle)
    iii. The cameras doesn't move
    These following assumptions aren't true for my use case:
    i. Players on the same team wear the same shirt colors
    ii. The goalkeepers remain the same throughout the game
    iii. The teams remain the same (I can let you know each time something changed)
    Search in the internet for something similar and come back with a plan on how to create it. Don't necesserily use what I started to do
    For each assumption you are using you will have to create an easy way to insert the data, and let me know what you are planning to use.
    Built in tracking/ - assumptions entered in matches/<game>/match.yaml plus tools.calibrate and tools.enroll.
    Open: learned ReID encoder (OSNet) instead of colour histograms, detector recall on far players, ball detection rate.
4. [ ] Please create an overlay video of what you made on a video in the internet that matches best to what I described
5. [ ] After detecting and tracking the players I want to create the following stats per player and per group:
    i. [ ] Possesion also per area (heatmaps)
    ii. [ ] Goals
    iii. [ ] Shots
    iv. [ ] Dangerous situations
    v. [ ] Corners
    vi. [ ] Succesful tackels
    vii. [ ] General score
