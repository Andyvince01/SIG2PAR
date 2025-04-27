''' 💥 src.utils.logger.py
>>> This module contains the Logger class which is used to set up a logger.
'''

#------------------------------------------------------------------------------------------------------------#
# 1. LIBRARIES AND MODULES
#------------------------------------------------------------------------------------------------------------#
import logging
import os
import platform

#--- Environment Booleans ---#
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans

#------------------------------------------------------------------------------------------------------------#
# 2. LOGGER CLASS
#------------------------------------------------------------------------------------------------------------#
class Logger:
    ''' This class is used to set up a logger. '''

    def __init__(self, log_name : str, log_level : int = logging.INFO, on_file : str = 'logs', on_screen : bool = True):
        ''' This class initializes a logger.
        
        Parameters
        ----------
        log_name : str
            The name of the logger.
        log_level : int, optional
            The logging level. The default is logging.INFO.
        on_file : str, optional
            The filename of the log. The default is 'log'. If None, the log is not saved to a file.
        on_screen : bool, optional
            Whether to log to the screen. The default is True. If False, the log is not displayed on the screen.
        '''
        #--- Set the logger attributes ---#
        self._log_name = log_name
        self._log_level = self._set_level(log_level)
        self._on_file = 'logs' if on_file is True else on_file
        self._on_screen = on_screen
                
        #--- Initialize the logger ---#
        self._logger = logging.getLogger(self._log_name)
        self._logger.propagate = False
        
        #--- Set up the logger ---#
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)                      
        
        #--- Set up the logger to log to a file and/or the screen ---#
        self._setup_log_file()
        self._setup_screen()

        #--- Set the logging level of the logger ---#
        self.logger.setLevel(self.log_level)
    
    #--- Public Functions ---#
    
    def color(self, *args) -> str:
        ''' This function returns the ANSI escape codes for the specified color and style.
            
        Parameters
        ----------
        args : tuple
            The input arguments. The first two arguments are the color and style, respectively. The last argument is the string to color.
        string : str, optional
            The input string. The default is "".
        
        Returns
        -------
        str
            The input string wrapped with ANSI escape codes for the specified color and style.
        '''
        #--- Return the ANSI escape codes for the specified color and style ---#
        return ANSI()(*args)
    
    def debug(self, message : str) -> None:
        ''' This function logs a debug message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to debug and color the message of green ---#
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(self.color('green', self._emojis(message)))

    def error(self, message : str) -> None:
        ''' This function logs an error message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to error and color the message of red ---#
        self.logger.setLevel(logging.ERROR)
        self.logger.error(self.color('red', 'underline', 'bold', self._emojis(message)))
    
    def info(self, message : str) -> None:
        ''' This function logs an info message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to info and color the message of black ---#
        self.logger.setLevel(logging.INFO)
        self.logger.info(self._emojis(message))
    
    def warning(self, message : str) -> None:
        ''' This function logs a warning message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to warning and color the message of yellow ---#
        self.logger.setLevel(logging.WARNING)
        self.logger.warning(self.color('yellow', 'underline', self._emojis(message)))

    #--- Private Functions ---#
    def _emojis(self, string : str = '') -> str:
        ''' This function a platform-dependent emoji-safe version of the input string.
        
        Parameters
        ----------
        string : str, optional
            The input string. The default is "".

        Returns
        -------
        str
            The string with emojis. 
        '''
        #--- Return the string with emojis if the platform is not Windows ---#
        return string.encode().decode("ascii", "ignore") if WINDOWS else string

    def _set_level(self, log_level : str = 'info') -> int:
        ''' This function sets the log_level of the logger.
        
        Parameters
        ----------
        log_level : str, optional
            The logging level. The default is 'info'.
        
        Returns
        -------
        int
            The logging level. If the log_level is not found, it returns logging.NOTSET (0).
        '''
        #--- Set the logging level ---#
        try:
            return getattr(logging, log_level.upper())
        except AttributeError:
            return logging.NOTSET                       # Return the default level - All messages are processed!

    def _setup_log_file(self) -> None:
        ''' This function sets up the log file. '''
        #--- Return if the log file is not set ---#
        if not self.on_file: return
        
        #--- Set the filename of the log file and create the directory if it does not exist ---#
        os.makedirs(self.on_file, exist_ok=True); self.logger.removeHandler(self.fh if hasattr(self, 'fh') else None)
        filename = os.path.join(self.on_file, f'{self._log_name}.log')

        #--- Set the file handler of the logger ---#
        self.fh = logging.FileHandler(filename=filename)
        
        formatter = logging.Formatter(f'\033[1m[%(asctime)s]\033[0m ({self._log_name}) - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.fh.setLevel(self.log_level); self.fh.setFormatter(formatter)

        #--- Add the file handler to the logger ---#
        self.logger.addHandler(self.fh)

    def _setup_screen(self) -> None:
        ''' This function sets up the screen. '''
        #--- Return if the screen is not set ---#
        if not self.on_screen: return
        
        #--- Remove the stream handler of the logger if it exists ---#
        self.logger.removeHandler(self.ch if hasattr(self, 'ch') else None)

        #--- Set the stream handler of the logger ---#
        self.ch = logging.StreamHandler()
        
        formatter = logging.Formatter(f'\033[1m[%(asctime)s]\033[0m ({self._log_name}) - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.ch.setLevel(self.log_level); self.ch.setFormatter(formatter)
        
        #--- Add the stream handler to the logger ---#
        self.logger.addHandler(self.ch)    
    
    #--- Helper Functions ---#
    def __str__(self) -> str:
        return f"Logger: {self._log_name}{' at' + self._on_file if self._on_file else ''} with level {self._log_level} and screen={self._on_screen}."
    
    #--- Setter and Getter Functions ---#
    @property
    def log_name(self) -> str:
        ''' The logger name. '''
        return self.log_name
    
    @log_name.setter
    def log_name(self, value : str) -> None:
        ''' The logger name setter. '''
        self._log_name = value.lower()
        if self.on_file: self._setup_log_file()
        if self.on_screen: self._setup_screen()
        
    @property
    def log_level(self) -> int:
        ''' The logger level. '''
        return self._log_level
    
    @log_level.setter
    def log_level(self, value : int) -> None:
        ''' The logger level setter. '''
        self._log_level = value
        if self.on_file: self._setup_log_file()
        if self.on_screen: self._setup_screen()
        
    @property
    def on_file(self) -> str:
        ''' The logger file. '''
        return self._on_file
    
    @on_file.setter
    def on_file(self, value : str) -> None:
        ''' The logger file setter. '''
        self._on_file = value

    @property
    def on_screen(self) -> bool:
        ''' The logger screen. '''
        return self._on_screen
    
    @on_screen.setter
    def on_screen(self, value : bool) -> None:
        ''' The logger screen setter. '''
        self._on_screen = value
        
    @property
    def logger(self) -> logging.Logger:
        ''' The logger. '''
        return self._logger
    
    @logger.setter
    def logger(self, value : logging.Logger) -> None:
        ''' The logger setter. '''
        self._logger = value

#------------------------------------------------------------------------------------------------------------#
# 3. ANSI CLASS
#------------------------------------------------------------------------------------------------------------#
class ANSI:
    ''' This class contains the ANSI escape codes for colors and styles. '''
    
    def __init__(self) -> None:
        ''' The constructor for the ANSI class. '''
        #--- Set the ANSI escape codes for colors and styles ---#
        self.ansi = {
            #--- Classic colors ---#
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            #--- Classic Background colors ---#
            "bg_black": "\033[40m",
            "bg_red": "\033[41m",
            "bg_green": "\033[42m",
            "bg_yellow": "\033[43m",
            "bg_blue": "\033[44m",
            "bg_magenta": "\033[45m",
            "bg_cyan": "\033[46m",
            "bg_white": "\033[47m",
            #--- Bright colors ---#
            "bright_black": "\033[90m",
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
            #--- Bright Background colors ---#
            "bg_bright_black": "\033[100m",
            "bg_bright_red": "\033[101m",
            "bg_bright_green": "\033[102m",
            "bg_bright_yellow": "\033[103m",
            "bg_bright_blue": "\033[104m",
            "bg_bright_magenta": "\033[105m",
            "bg_bright_cyan": "\033[106m",
            "bg_bright_white": "\033[107m",   
            #--- Styles ---#
            "end": "\033[0m",
            "bold": "\033[1m",
            "italic": "\033[3m",
            "underline": "\033[4m",
        }

    def __call__(self, *args) -> str:
        ''' This function returns the ANSI escape codes for the specified color and style.
        
        Parameters
        ----------
        args : tuple
            The input arguments. The first two arguments are the color and style, respectively. The last argument is the string to color.
        string : str, optional
            The input string. The default is "".
        
        Returns
        -------
        str
            The input string wrapped with ANSI escape codes for the specified color and style.
        '''
        #--- Return the input string if there is only one argument ---#
        if len(args) == 1: return args[0]
        
        #--- Get the color and style ---#
        *styles, string = args
        
        #--- Return the ANSI escape codes for the specified color and style ---#
        return "".join(self.ansi[x] for x in styles) + f"{string}" + self.ansi["end"]

if __name__ == '__main__':
    #--- Test the Logger Class ---#
    logger = Logger(log_name='base', log_level='Invalid', on_file=True, on_screen=True)
    #--- Test the Logger Class ---#
    logger.info('Initial Dataset Finished 🎉.')
    logger.debug('Initial Model Finished 🎉.')
    logger.error('Begin Model Inference ❌.')