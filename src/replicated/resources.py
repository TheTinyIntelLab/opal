"""Process resource evidence with explicit scope and optional measurement."""
import sys


def memory_evidence():
    try:
        import psutil
        info=psutil.Process().memory_info()
        if hasattr(info,'peak_wset'):
            return {'peak_memory_bytes':int(info.peak_wset),
                    'memory_scope':'process lifetime peak working set; includes interpreter/dependencies',
                    'measurement_method':'psutil Windows peak_wset'}
        rss=int(info.rss)
    except ImportError:
        rss=None
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'peak_memory_bytes':int(peak*(1 if sys.platform=='darwin' else 1024)),
                'memory_scope':'process lifetime peak RSS; includes interpreter/dependencies',
                'measurement_method':'resource.ru_maxrss'}
    except ImportError:
        return {'peak_memory_bytes':None,'completion_rss_bytes':rss,
                'memory_scope':'peak unavailable; completion RSS is not a measured peak',
                'measurement_method':'psutil completion RSS' if rss is not None else 'unavailable'}
