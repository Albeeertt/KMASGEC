import subprocess
import logging

class Agat:
    def __init__(self, entorno: str):
        self._logger = logging.getLogger(__name__)
        self.entorno = entorno

    def add_introns(self, gff: str, route_out: str):
        self._logger.info("Añadiendo intrones.")
        name_out: str = "introns_add.gff3"
        cmd = ["conda", 
            "run", 
            "-n",
            self.entorno,
            "agat_sp_add_introns.pl",
            "-gff",
            gff,
            "--out",
            route_out+name_out
        ]
        cmd_str = ' '.join(cmd) + ' > /dev/null 2>&1'
        subprocess.run(cmd_str, check=True, shell=True)
        return route_out+name_out

    def add_intergenicRegion(self, gff: str, route_out: str):
        self._logger.info("Añadiendo regiones intergénicas.")
        name_out: str = "intergenic_regions_add.gff3"
        cmd = ["conda", 
            "run", 
            "-n",
            self.entorno,
            "agat_sp_add_intergenic_regions.pl",
            "-gff",
            gff,
            "--out",
            route_out+name_out
        ]
        cmd_str = ' '.join(cmd) + ' > /dev/null 2>&1'
        subprocess.run(cmd_str, shell=True, check=True)
        return route_out+name_out
    
    def keep_longest_isoform(self, gff: str, route_out: str):
        self._logger.info("Manteniendo la isoforma más grande.")
        name_out: str = "keep_longest_isoform.gff3"
        cmd = ["conda", 
            "run", 
            "-n",
            self.entorno,
            "agat_sp_keep_longest_isoform.pl",
            "-gff",
            gff,
            "--out",
            route_out+name_out
        ]
        cmd_str = ' '.join(cmd) + ' > /dev/null 2>&1'
        subprocess.run(cmd_str, shell=True, check=True)
        return route_out+name_out
        
        
    def good_format(self, gff: str, route_out: str):
        self._logger.info("Formato correcto.")
        name_out: str = "goodFormat.gff3"
        cmd = ["conda",
            "run",
            "-n",
            self.entorno,
            "agat_convert_sp_gxf2gxf.pl",
            "-g",
            gff,
            "-o",
            route_out+name_out
        ]
        cmd_str = ' '.join(cmd) + ' > /dev/null 2>&1'
        subprocess.run(cmd_str, shell=True, check=True)
        return route_out+name_out
