# BullyGuard

A Discord bot that I created to catch any toxic/bullying messages on our gaming server. The model at the backend is a derivative of the DistilBERT model fine-tuned on the cyberbullying detection task. 

Quite simple - bot catches toxic stuff, and instead of just flagging stuff, it's got this whole escalation system going on. First time? Just a friendly DM. Second time? A timeout to think about life. Third time? Timeout span increases, well it goes on. 

## Features 

- Does what it says - catches toxic messages
- Keeps the mods in the loop with a dedicated channel for all the tea
- Has this whole "three strikes" system that actually works
- Used ONNX (coz it's fast)
- Some other stuff I'll probably add here if I remember

## How I built this 
So I started with [this](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset) dataset from Kaggle - it has multiple files which I squashed together and added a private dataset of mine on top. You can find the complete dataset [here](https://www.kaggle.com/datasets/hawkeye2704/cyberbullying-dataset). It's quite extensive with labels saying if they're toxic or not. About 15% are toxic which is pretty much what you'd expect in real life.
### The Model
- Went with DistilBERT because:
  - BERT was overkill
  - Needed something fast for real-time checks
  - Still smart enough to catch the nasty stuff
- Added my own model architecture (simple one tbf!) downstream to the DistilBERT
- Used this new cool optimizer called ADOPT instead of the usual Adam (you gotta try it) [ADOPT GitHub](https://github.com/iShohei220/adopt) [ADOPT Paper](https://arxiv.org/abs/2411.02853)

### How Good Is It?
I'd call it good enough to be honest, I did experiment a lot with the dataset and the model architecture.
- Has a recall of 0.77 on the target class
- Recall is the focus because I'd rather catch a few extra than miss the bad ones
- Accuracy was 0.90 given the number of non-toxic samples being identified right, but I wouldn't mind it

If you're into the technical details, check out my [notebook](notebooks/Aggression_Detection_Rework.ipynb). It is probably the cleanest notebook I've ever cooked.

### ONNX for inference
Converted the whole thing to ONNX because:
- It's faster (like, way faster)
- Uses less memory
- Just works better for what we're doing

You can grab the model from HuggingFace if you wanna play with it. I've got it set up so the bot pulls it automatically. [BullyGuard](https://huggingface.co/karthikarunr/BullyGuard)

## Bot In Action 
Alright, so you probably wanna see this thing actually working. I've got some videos for you to take a peek.

https://github.com/user-attachments/assets/001aa048-3756-4ff9-9c35-8e867c9c5348

https://github.com/user-attachments/assets/32b5c33b-e7a4-44e0-8aa7-c929b977b845

https://github.com/user-attachments/assets/5a48a042-1676-4bb0-a003-597265416ce6

Same incident but from the mod channel - see all the reporting real-time.

![Desktop Screenshot 2024 11 20 - 15 57 04 20](https://github.com/user-attachments/assets/9032cb2d-1bed-4726-9f7c-3b4e56f40be3)

### What's Actually Happening (just a text version of the above demos)
When someone sends a toxic message:
- Bot catches it (with the model at the backend)
- Mods get all the details in a dedicated channel
- The troublemaker gets a series of increasingly annoyed responses:
  - First time: A warning
  - Second time: "Time to chill for 10 seconds"
  - Third time: "Okay, now you get 30 seconds to think about what you did"
And yeah, it keeps going (obviously these rules can ease out or tighten as per setting in the code)

### Deployment on AWS 
So I've got this running on EC2 and honestly, the free tier is pretty sweet for this kind of project. 
Here's what I'm using:
- t2.micro instance (because it's free and the bot isn't that hungry)
- Amazon Linux 2023 (I did try Ubuntu 22.04 but it wasn't a good experience)
- 16GB storage (might give you an issue during torch install but can work around)

What It Costs
- EC2: $0 (thanks, free tier)
- Storage: $0 (under the 30GB limit)

The Actual Deployment
- Got it running with systemd
- Auto-starts if the instance reboots

Just make sure to set up CloudWatch Billing alerts to get notified before AWS decides to dunk a big bill on your side project. 

## Running Your Own BullyGuard 

I'd love to make this a public bot, but:
- I need the t2.micro for my other projects
- Each server needs its own mod channel setup
- You probably want your own timeout rules / ban rules

So here's how you can run your own version instead:
### Quick Setup

**1. Environment Setup**

  - First, get your Python environment ready:
    - Make sure you have Python 3.10+ installed
    - Clone this repository
    - Create a virtual environment (I use [uv](https://docs.astral.sh/uv/) for managing Python packages, but I'll leave it to you)
      - ```bash
        python -m venv venv 
        ```
      - ```bash
        source venv/bin/activate (Linux/Mac)
        ```
      - ```bash
        venv\Scripts\activate (Windows)
        ```
    - Install dependencies: 
        ```bash
        pip install -r requirements.txt
        ```
**2. Discord Bot Creation**

  - Head over to Discord Developer Portal (https://discord.com/developers/applications):

    - Click "New Application"
    - Name it
    - Go to "Bot" section
    - Click "Reset Token" and save that token (you'll need it later)
    - Enable these "Privileged Gateway Intents":
        - Message Content Intent
        - Server Members Intent
        - Presence Intent

**3. Bot Permissions**

  - Your bot needs some powers:
    - In the Developer Portal, go to "OAuth2" → "URL Generator"
    - Select these scopes:
        - bot
        - applications.commands
    - Select these bot permissions:
        - Moderate Members (for timeouts)
        - Send Messages
        - Read Messages/View Channels
        - Read Message History
        - Use Slash Commands
        - Copy the generated URL
        - Open it in a browser
        - Select your server
        - Authorize the bot

**4. Server Setup**

  - In your Discord server:
    - Create a channel for mod logs (like #moderator-logs)
    - Right click → Copy Channel ID
    - Make sure bot's role is ABOVE the roles it needs to moderate

**5. Configuration**

  - Create a .env file in the project root:
```yml
DISCORD_BOT_TOKEN=your-bot-token-here
MOD_CHANNEL_ID=your-mod-channel-id
```

  - The config/config.yaml has some defaults:
```yaml
toxicity_threshold: 0.75  # Higher = less sensitive
warning_threshold: 3      # Strikes before timeout
cache_size: 1000         # Increase if you have RAM to spare
```
**6. Running the Bot**

  - Activate your virtual environment (if not already active)
    ```bash
    python run.py
    ```
  - You should see "Bot is ready!" and your bot should be online

**7. Testing**

  - Try these to make sure everything works:
    - /check command with a test message
    - Send a toxic message (bot should catch it)
    - Check your mod channel for logs

  - Common Issues
    - Bot not responding: Check if all intents are enabled
    - Can't timeout: Check bot's role position in server
    - No mod logs: Verify channel ID and bot's channel permissions

That's it! Your bot should now be online on your server. Want to run it 24/7 on AWS? Check out the AWS Deployment section above.

## Credits
For the ADOPT optimizer
```text
@inproceedings{taniguchi2024adopt,
 author={Taniguchi, Shohei and Harada, Keno and Minegishi, Gouki and Oshima, Yuta and Jeong, Seong Cheol and Nagahara, Go and Iiyama, Tomoshi and Suzuki, Masahiro and Iwasawa, Yusuke and Matsuo, Yutaka},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate},
 year = {2024}
}
```

## License
[Apache 2.0](./LICENSE)

## Contact
If you wanna talk over this repo or any of my work or click some heads in Counter-Strike:

- Discord: hawkeye2704
- Email: karthikarun2000@gmail.com
