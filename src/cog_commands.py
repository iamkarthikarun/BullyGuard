import os

import discord
from discord import app_commands
from discord.ext import commands

from .model_handler import ToxicityPredictor
from .utils import BotConfig, MessageFormatter, ModLogger


class ModerationCommands(commands.Cog):

    """Implements command handling for toxicity detection and moderation.

    This Cog provides slash commands for message checking, user history viewing,
    and bot configuration, along with automatic message monitoring.

    Attributes:
        bot: The Discord bot instance.
        predictor: Toxicity prediction model handler.
        logger: Moderation action logger.
        formatter: Message formatting utility.
        config: Bot configuration manager.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the moderation commands cog.

        Args:
            bot: Discord bot instance to attach commands to.

        Raises:
            ValueError: If MOD_CHANNEL_ID environment variable is not set.
        """
        self.bot = bot
        self.predictor = ToxicityPredictor('models/cyberbullying_model.onnx')
        self.logger = ModLogger(mod_channel_id=int(os.getenv('MOD_CHANNEL_ID', 0)))
        self.formatter = MessageFormatter()
        self.config = BotConfig()

    @app_commands.command(name="check")
    async def check_message(self, interaction: discord.Interaction, message: str) -> None:
        """Analyzes a message for toxic content.

        Args:
            interaction: Discord interaction context.
            message: The message text to analyze.

        Raises:
            Exception: If message analysis fails.
        """
        await interaction.response.defer()
        
        try:
            is_toxic, confidence = self.predictor.predict(message)
            response = self.formatter.format_toxicity_report(
                message, is_toxic, confidence)
            await interaction.followup.send(response)
            
        except Exception as e:
            await interaction.followup.send(f'Error analyzing message: {str(e)}')

    @app_commands.command(name="history")
    @app_commands.default_permissions(manage_messages=True)

    async def get_history(self, interaction: discord.Interaction, user: discord.User) -> None:
        """Retrieves moderation history for a specified user.

        Args:
            interaction: Discord interaction context.
            user: The Discord user whose history to retrieve.
        """
        await interaction.response.defer()
        
        history = self.logger.get_user_history(user.id)
        response = self.formatter.format_user_history(history)
        await interaction.followup.send(response)

    @app_commands.command(name="threshold")
    @app_commands.default_permissions(administrator=True)

    async def set_threshold(self, interaction: discord.Interaction,value: float) -> None:
        """Sets the toxicity detection sensitivity threshold.

        Args:
            interaction: Discord interaction context.
            value: New threshold value between 0 and 1.
        """
        if not 0 <= value <= 1:
            await interaction.response.send_message('Threshold must be between 0 and 1')
            return
            
        self.config.set('toxicity_threshold', value)
        await interaction.response.send_message(
            f'Toxicity threshold set to {value}')

    @app_commands.command(name="stats")
    async def get_stats(self, interaction: discord.Interaction) -> None:
        """Displays current bot statistics and configuration.

        Args:
            interaction: Discord interaction context.
        """
        cache_stats = self.predictor.get_cache_stats()
        threshold = self.config.get('toxicity_threshold')
        
        embed = discord.Embed(title='Bot Statistics', color=discord.Color.blue())
        embed.add_field(name='Cache Usage', value=f"{cache_stats['cache_size']}/{cache_stats['cache_limit']}")
        embed.add_field(name='Toxicity Threshold', value=f'{threshold:.2f}')
        
        await interaction.response.send_message(embed=embed)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Monitors messages for toxic content.

        Automatically analyzes new messages for toxicity and takes appropriate
        moderation actions based on configured thresholds.

        Args:
            message: The Discord message to analyze.
        """
        if message.author.bot:
            return
                
        try:
            is_toxic, confidence = self.predictor.predict(message.content)
            threshold = self.config.get('toxicity_threshold')
                
            if is_toxic and confidence > threshold:
                await self.logger.handle_violation(
                    user=message.author,
                    message=message,
                    confidence=confidence * 100,
                    bot=self.bot
                )
                    
        except Exception as e:
            print(f'Error processing message: {str(e)}')

async def setup(bot: commands.Bot) -> None:
    """Adds the moderation commands cog to the bot.

    Args:
        bot: The Discord bot instance to add commands to.
    """
    await bot.add_cog(ModerationCommands(bot))
