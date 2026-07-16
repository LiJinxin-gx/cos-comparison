from pathlib import Path

package_dir = Path(__file__).parent.resolve()
__path__.insert(0, str(package_dir.parent))

import logic
import refection
