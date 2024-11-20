import asyncio
import os
import traceback
from typing import Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv


class ToxicityBot(commands.Bot):

    """Main bot class implementing toxicity detection and moderation.

    This class extends Discord's Bot class to provide toxicity detection
    capabilities, command handling, and event management.

    Attributes:
        startup_time: DateTime when the bot successfully connected to Discord.
    """

    def __init__(self) -> None:
        """Initializes the bot with required intents and settings.

        The bot is configured with message content and member intents enabled,
        and the default help command is disabled in favor of custom help.
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        super().__init__(
            command_prefix='!',  # Fallback prefix for regular commands
            intents=intents,
            help_command=None  # Disable default help command
        )
        
        self.startup_time: Optional[discord.utils.UTCDateTime] = None
        
    async def setup_hook(self) -> None:
        """Performs pre-startup initialization.

        Loads command cogs and synchronizes the command tree with Discord.
        This method is called automatically before the bot starts.

        Raises:
            ExtensionNotFound: If the cog module cannot be found.
            ExtensionFailed: If the cog fails to load.
        """
        await self.load_extension('src.cog_commands')
        
        print('Syncing command tree...')
        await self.tree.sync()
        print('Command tree synced!')
        
    async def on_ready(self) -> None:
        """Handles bot ready event.

        Sets up the bot's presence and records startup time.
        This method is called when the bot has successfully connected to Discord.
        """
        self.startup_time = discord.utils.utcnow()
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')
        
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name='for toxic messages'
            )
        )
    
    async def on_error(self, event_method: str) -> None:
        """Global error handler for all bot events.

        Args:
            event_method: Name of the event method that raised the error.
            *args: Positional arguments passed to the event method.
            **kwargs: Keyword arguments passed to the event method.
        """
        print(f'Error in {event_method}:')
        traceback.print_exc()

async def main() -> None:
    """Initializes and starts the bot.

    This function handles the bot's startup sequence, including environment
    variable loading and error handling.

    Raises:
        ValueError: If the Discord token is not found in environment variables.
        Exception: If the bot encounters an error during startup.
    """
    load_dotenv()
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        raise ValueError('No Discord token found in environment variables!')
    
    bot = ToxicityBot()
    
    try:
        print('Starting bot...')
        await bot.start(token)
    except KeyboardInterrupt:
        print('\nShutting down...')
        await bot.close()
    except Exception as e:
        print(f'Error: {str(e)}')
        await bot.close()

if __name__ == '__main__':
    asyncio.run(main())
