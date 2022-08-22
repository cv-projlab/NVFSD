
def empty_decorator(fn, **_):
   def decorate(*args, **kwargs):
      return fn(*args, **kwargs)
   return decorate

# TRY IMPORT EMAIL-TOOLS
try:
   from email_tools import email_notification_wrapper
   error_notif = lambda yes: email_notification_wrapper if yes else empty_decorator
except ImportError:
   print(f'Package: email-tools not installed ! Use < pip install git+https://github.com/AndreGraca98/email-tools.git > to install package')
   error_notif = lambda yes: empty_decorator


# ENDFILE
