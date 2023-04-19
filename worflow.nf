#!/usr/bin/env nextflow

params.configfile = "config.yaml"

// Read configuration file
config = yaml(params.configfile)

