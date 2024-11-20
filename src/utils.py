import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import discord
import yaml
from discord.ext import commands


class ModLogger:

    """Handles logging and management of moderation actions.

    This class manages the logging of moderation actions, tracks user violations,
    and implements an escalating response system for toxic message detection.

    Attributes:
        log_dir: Path object pointing to the log directory.
        current_log_file: Path object for the current month's log file.
        mod_channel_id: Discord channel ID for moderation notifications.
        user_violation_count: Counter tracking violations per user.
    """

    def __init__(self, log_dir: str = 'logs', mod_channel_id: Optional[int] = None) -> None:
        """Initializes the moderation logging system.

        Args:
            log_dir: Directory path where log files will be stored.
            mod_channel_id: Discord channel ID for sending mod notifications.

        Raises:
            PermissionError: If log directory cannot be created.
        """

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_log_file = self.log_dir / f"mod_log_{datetime.datetime.now().strftime('%Y%m')}.json"
        self.mod_channel_id = mod_channel_id
        self.user_violation_count = defaultdict(int)
        
    def log_action(self, action_type: str, **kwargs: Any) -> Dict[str, Any]:
        """Logs a moderation action with timestamp and details.

        Args:
            action_type: Category of moderation action (e.g., 'message_deleted').
            **kwargs: Additional details about the action to be logged.

        Returns:
            Dictionary containing the complete logged action details.

        Raises:
            IOError: If writing to log file fails.
        """

        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action_type,
            **kwargs
        }
        
        logs = self._read_logs()
        logs.append(log_entry)
        self._write_logs(logs)
        
        return log_entry
    
    def get_user_history(self, user_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves moderation history for a specific user.

        Args:
            user_id: Discord user ID to get history for.
            limit: Maximum number of entries to return (newest first).

        Returns:
            List of moderation actions associated with the user.
        """

        logs = self._read_logs()
        user_logs = [log for log in logs if log.get('user_id') == user_id]
        
        if limit:
            user_logs = user_logs[-limit:]
            
        return user_logs
    
    def _read_logs(self) -> List[Dict[str, Any]]:
        """Reads existing logs from the current log file.

        Returns:
            List of log entries from the current log file.
        """

        if not self.current_log_file.exists():
            return []
            
        try:
            with open(self.current_log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    
    def _write_logs(self, logs: List[Dict[str, Any]]) -> None:
        """Writes log entries to the current log file.

        Args:
            logs: List of log entries to write.

        Raises:
            IOError: If writing to the log file fails.
        """

        with open(self.current_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
    
    async def handle_violation(self, user: discord.Member, message: discord.Message, confidence: float, bot: commands.Bot) -> None:
        """Handles toxic message violations with escalating consequences.

        Implements a three-strike system with increasing timeout durations
        and sends notifications to both the user and moderation channel.

        Args:
            user: Discord member who sent the toxic message.
            message: The toxic message that was detected.
            confidence: Toxicity detection confidence score (0-100).
            bot: Discord bot instance for channel access.

        Raises:
            discord.Forbidden: If bot lacks permissions for timeout or DM.
        """

        self.user_violation_count[user.id] += 1
        violation_count = self.user_violation_count[user.id]
        
        mod_channel = bot.get_channel(self.mod_channel_id)
        if mod_channel:
            await mod_channel.send(
                f'🚨 **Toxic Message Detected**\n'
                f'User: {user.mention} ({user.name})\n'
                f'Channel: {message.channel.mention}\n'
                f'Confidence: {confidence:.2f}%\n'
                f'Violation #{violation_count}\n'
                f'Message: ```{message.content}```'
            )

        try:
            if violation_count == 1:
                await user.send(
                    f'⚠️ **Warning**: Your message in {message.channel.mention} '
                    f'was flagged as toxic.\nThis is your first warning. Please '
                    f'be mindful of our community guidelines.'
                )
            
            elif violation_count == 2:
                await user.timeout(
                    datetime.timedelta(seconds=10),
                    reason='Second toxic message violation'
                )
                await user.send(
                    f'⚠️ Your message in {message.channel.mention} was flagged '
                    f'as toxic.\nAs this is your second violation, you have been '
                    f'timed out for 10 seconds.'
                )
            
            else:
                await user.timeout(
                    datetime.timedelta(seconds=30),
                    reason=f'Toxic message violation #{violation_count}'
                )
                await user.send(
                    f'⚠️ Your message in {message.channel.mention} was flagged '
                    f'as toxic.\nAs this is your {violation_count}rd violation, '
                    f'you have been timed out for 30 seconds.'
                )

        except discord.Forbidden:
            if mod_channel:
                await mod_channel.send(
                    f'⚠️ Failed to handle consequences for {user.mention}. '
                    f'Please check bot permissions.'
                )

        self.log_action(
            action_type='toxic_message_detected',
            user_id=user.id,
            content=message.content,
            confidence=confidence
        )

class MessageFormatter:
    """Handles formatting of Discord messages for toxicity reports and user history."""

    @staticmethod
    def format_toxicity_report(message_content: str, is_toxic: bool, confidence: float) -> str:
        """Formats toxicity analysis results for Discord display.

        Args:
            message_content: The message that was analyzed.
            is_toxic: Whether the message was classified as toxic.
            confidence: Model's confidence score (0-1).

        Returns:
            A formatted string suitable for Discord message display.
        """

        status = '🔴 Toxic' if is_toxic else '🟢 Non-toxic'
        confidence_percent = round(confidence * 100, 2)
        
        return (
            f'**Message Analysis**\n'
            f'Status: {status}\n'
            f'Confidence: {confidence_percent}%\n'
            f'```{message_content}```'
        )
    
    @staticmethod
    def format_user_history(history: List[Dict[str, Any]]) -> str:
        """Formats user moderation history for Discord display.

        Args:
            history: List of moderation action dictionaries.

        Returns:
            A formatted string containing the user's moderation history.
        """
        
        if not history:
            return 'No moderation history found.'
            
        formatted = ['**Moderation History**']
        for entry in history:
            timestamp = datetime.datetime.fromisoformat(
                entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            formatted.append(
                f"• {timestamp} - {entry['action'].replace('_', ' ').title()}\n"
                f"  Reason: {entry.get('reason', 'No reason provided')}"
            )
            
        return '\n'.join(formatted)

class BotConfig:
    """Manages bot configuration through YAML files.

    This class handles loading, saving, and accessing bot configuration settings,
    with support for default values and persistent storage.

    Attributes:
        config_file: Path to the YAML configuration file.
        defaults: Dictionary of default configuration values.
        config: Dictionary of current configuration values.
    """

    def __init__(self, config_file: str = 'config/config.yaml') -> None:
        """Initializes the configuration manager.

        Args:
            config_file: Path to the YAML configuration file.

        Raises:
            PermissionError: If config directory cannot be created/accessed.
        """
        self.config_file = Path(config_file)
        self.defaults: Dict[str, Union[float, int, None]] = {
            'toxicity_threshold': 0.5,
            'warning_threshold': 3,
            'cache_size': 1000,
            'log_channel': None
        }
        self.config = self._load_config()
    
    def get(self, key: str) -> Union[float, int, None]:
        """Retrieves a configuration value.

        Args:
            key: Configuration key to retrieve.

        Returns:
            The configuration value, or the default value if not found.
        """
        return self.config.get(key, self.defaults.get(key))
    
    def set(self, key: str, value: Union[float, int, None]) -> None:
        """Updates a configuration value.

        Args:
            key: Configuration key to update.
            value: New value to set.

        Raises:
            IOError: If saving the configuration fails.
        """
        self.config[key] = value
        self._save_config()
    
    def _load_config(self) -> Dict[str, Union[float, int, None]]:
        """Loads configuration from YAML file.

        Returns:
            Dictionary containing configuration values.
        """
        if not self.config_file.exists():
            return self.defaults.copy()
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return {**self.defaults, **yaml.safe_load(f)}
        except Exception:
            return self.defaults.copy()
    
    def _save_config(self) -> None:
        """Saves current configuration to YAML file.

        Raises:
            IOError: If writing to the configuration file fails.
            PermissionError: If config directory cannot be created.
        """
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f)
