import subprocess
import os
import sys
import logging
import argparse
import shutil
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TOP_LEVEL_SIMULATION_OUTPUTS = (
    "NDXMEGASIM.csv",
    "NDXMEGAPRICESIM.csv",
    "NDXMEGA2SIM.csv",
    "NDX30SIM.csv",
)

def run_script(script_path, desc, env=None, script_args=None):
    """Run a Python script with optional environment variables."""
    logging.info(f"--- Starting: {desc} ---")
    try:
        # verify file exists
        if not os.path.exists(script_path):
             logging.error(f"Script not found: {script_path}")
             sys.exit(1)

        # Merge with current environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Run with the current interpreter so virtualenv/module resolution stays consistent.
        command = [sys.executable, script_path]
        if script_args:
            command.extend(script_args)
        subprocess.run(command, check=True, text=True, env=run_env)
        logging.info(f"--- Completed: {desc} ---\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"!!! Failed: {desc} (Exit Code: {e.returncode}) !!!")
        sys.exit(e.returncode)


def run_ndxmega_backtests(script_path, env_vars):
    """Build both return conventions consumed by NDXMEGA and QQUPSIM.

    The price-return run must happen first because both modes write shared
    diagnostic charts and constituent reports. Running total return last keeps
    those shared artifacts aligned with the primary NDXMEGASIM output.
    """
    price_return_env = {**env_vars, "NDX_PRICE_RETURN": "1"}
    run_script(
        script_path,
        "Backtest NDX Mega 1.0 Price Return (QQUPSIM)",
        price_return_env,
    )

    total_return_env = {**env_vars, "NDX_PRICE_RETURN": "0"}
    run_script(
        script_path,
        "Backtest NDX Mega 1.0 Total Return",
        total_return_env,
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Rebuild NDX/NDXMEGA/NDXMEGA2/NDX30 simulation pipeline.',
        epilog='''
Examples:
  python rebuild_all.py                                       # Use yfinance (default)
  python rebuild_all.py --refresh-official-membership         # Also refresh archived Nasdaq membership
  python rebuild_all.py --stooq                               # Use Stooq (free, 20+ years)
  python rebuild_all.py --polygon YOUR_API_KEY                # Use Polygon.io (paid)
        '''
    )
    parser.add_argument(
        '--polygon', 
        metavar='API_KEY',
        help='Use Polygon.io as data source with the provided API key (paid)'
    )
    parser.add_argument(
        '--stooq',
        action='store_true',
        help='Use Stooq as data source (free, 20+ years of data)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip SEC download. Use existing parsed components, or parse cached filings if needed.'
    )
    parser.add_argument(
        '--skip-reconstruct',
        action='store_true',
        help='Skip weight reconstruction step (use existing weights)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='DEEP CLEAN: Deletes all downloaded data, caches, and results before running.'
    )
    parser.add_argument(
        '--refresh-official-membership',
        action='store_true',
        help='Refresh the archived Nasdaq official membership snapshots before reconstruction.'
    )
    parser.add_argument(
        '--refresh-index-funds',
        action='store_true',
        help='Refresh free SEC filings from historical NDX-tracking funds.'
    )
    parser.add_argument(
        '--refresh-free-history',
        action='store_true',
        help='Refresh the public WIKI and CMU delisted-price archives.'
    )
    args = parser.parse_args()

    # Build environment variables to pass to child scripts
    env_vars = {}
    data_source_label = "yfinance"
    
    if args.polygon:
        env_vars['NDX_DATA_SOURCE'] = 'polygon'
        env_vars['POLYGON_API_KEY'] = args.polygon
        data_source_label = "Polygon.io"
        logging.info(f"Data Source: Polygon.io (paid)")
    elif args.stooq:
        env_vars['NDX_DATA_SOURCE'] = 'stooq'
        data_source_label = "Stooq"
        logging.info(f"Data Source: Stooq (free)")
    else:
        env_vars['NDX_DATA_SOURCE'] = 'yfinance'
        logging.info(f"Data Source: yfinance (default)")
    
    # Base paths
    # This script is in data/ndx_simulation/scripts/
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    module_root = os.path.dirname(scripts_dir)
    src_dir = os.path.join(module_root, "src")
    
    # Paths from config (replicated here or imported? safer to replicate relative to root to avoid ImportErrors if config moves)
    # But we can assume module structure.
    data_dir = os.path.join(module_root, "data")
    cache_dir = os.path.join(data_dir, "cache")
    assets_dir = os.path.join(data_dir, "assets")
    results_dir = os.path.join(data_dir, "results")
    charts_dir = os.path.join(results_dir, "charts")
    filings_dir = os.path.join(cache_dir, "ndx_filings")
    comp_file = os.path.join(assets_dir, "nasdaq_components.csv")
    weights_file = os.path.join(results_dir, "nasdaq_quarterly_weights.csv")
    parser_script = os.path.join(src_dir, "ndx_parser.py")
    
    if args.clear_cache:
        logging.warning("!!! --clear-cache SET: Performing Deep Clean !!!")
        logging.info(f"Cleaning {data_dir} artifacts...")
        
        # 1. Price Caches
        for pattern in ["prices_cache.pkl", "prices_cache_*.pkl"]:
            for f in glob.glob(os.path.join(cache_dir, pattern)):
                try:
                    os.remove(f)
                    logging.info(f"Deleted: {f}")
                except OSError as e:
                    logging.error(f"Error deleting {f}: {e}")
                    
        # 2. NDX Filings (Downloads)
        if os.path.exists(filings_dir):
            try:
                shutil.rmtree(filings_dir)
                logging.info(f"Deleted Directory: {filings_dir}")
            except OSError as e:
                logging.error(f"Error deleting {filings_dir}: {e}")
        
        # 3. Assets (Components CSV)
        if os.path.exists(comp_file):
            try:
                os.remove(comp_file)
                logging.info(f"Deleted: {comp_file}")
            except OSError as e:
                logging.error(f"Error deleting {comp_file}: {e}")

        # 4. Results (Weights)
        if os.path.exists(weights_file):
            try:
                os.remove(weights_file)
                logging.info(f"Deleted: {weights_file}")
            except OSError as e:
                logging.error(f"Error deleting {weights_file}: {e}")
        
        # 5. Charts
        if os.path.exists(charts_dir):
            # Delete all pngs
            for f in glob.glob(os.path.join(charts_dir, "*.png")):
                try:
                    os.remove(f)
                    logging.info(f"Deleted: {f}")
                except OSError as e:
                     logging.error(f"Error deleting {f}: {e}")
                     
        # 6. Top Level Simulation Outputs (NDXMEGASIM.csv)
        # Assuming they are in data/ndx_simulation/.. (based on backtest scripts)
        # ../NDXMEGASIM.csv relative to module_root
        parent_dir = os.path.dirname(module_root) # Testfol-MarginStresser/data usually? No, module_root is data/ndx_simulation. Parent is data.
        # Actually backtest script uses: os.path.join(config.BASE_DIR, "..", "NDXMEGASIM.csv")
        # config.BASE_DIR is data/ndx_simulation
        # So it is in Testfol-MarginStresser/data/NDXMEGASIM.csv (if that's where module is)
        # Let's rely on relative path logic matching backtest
        
        for sim_file in TOP_LEVEL_SIMULATION_OUTPUTS:
            p = os.path.abspath(os.path.join(module_root, "..", sim_file))
            if os.path.exists(p):
                 try:
                    os.remove(p)
                    logging.info(f"Deleted: {p}")
                 except OSError as e:
                    logging.error(f"Error deleting {p}: {e}")
                    
        logging.info("--- Deep Clean Completed ---\n")
    
    logging.info("Starting Full NDX Simulation Rebuild...")
    
    # 1. Download Filings (ndx_downloader.py is in src/)
    if not args.skip_download:
        downloader_script = os.path.join(src_dir, "ndx_downloader.py")
        run_script(downloader_script, "Download SEC Filings", env_vars)
        
        # 1.5. Parse Filings (ndx_parser.py is in src/)
        # Must run after download to generate nasdaq_components.csv
        run_script(parser_script, "Parse SEC Filings", env_vars)
    else:
        if os.path.exists(comp_file):
            logging.info(f"--- Skipped: Download SEC Filings (using existing parsed components at {comp_file}) ---\n")
        elif os.path.isdir(filings_dir) and any(
            name.endswith((".txt", ".htm", ".html")) for name in os.listdir(filings_dir)
        ):
            logging.info("--- Skipped: Download SEC Filings (parsing cached filings because components CSV is missing) ---")
            run_script(parser_script, "Parse Cached SEC Filings", env_vars)
        else:
            logging.error(
                "Cannot use --skip-download: no parsed components CSV and no cached SEC filings were found."
            )
            logging.error("Run without --skip-download, or restore data/ndx_simulation/data/cache/ndx_filings.")
            sys.exit(1)
    
    # 2. Update Name Mappings (src/mapper.py)
    # This ensures new filings are mapped to tickers before reconstruction
    mapper_script = os.path.join(src_dir, "mapper.py")
    run_script(mapper_script, "Update Name Mappings", env_vars)

    # 2.5. Refresh archived official Nasdaq membership snapshots used by backtests
    # Default is to skip this slow step unless explicitly requested.
    if args.refresh_official_membership:
        official_membership_script = os.path.join(scripts_dir, "download_official_membership.py")
        run_script(official_membership_script, "Refresh Official Nasdaq Membership Archive", env_vars)
    else:
        logging.info("--- Skipped: Refresh Official Nasdaq Membership Archive (pass --refresh-official-membership to enable) ---\n")

    # 2.75. Resolve dated annual and quarterly SEC holdings to the official
    # historical ticker roster. Reconstruction consumes this canonical layer.
    index_fund_script = os.path.join(src_dir, "sec_index_fund_holdings.py")
    index_fund_positions = os.path.join(assets_dir, "ndx_index_fund_positions.csv")
    if args.refresh_index_funds or not os.path.exists(index_fund_positions):
        refresh_args = ["--refresh"] if args.refresh_index_funds else []
        run_script(
            index_fund_script,
            "Build Historical SEC Index-Fund Holdings",
            env_vars,
            refresh_args,
        )
    else:
        logging.info("--- Reused: Cached Historical SEC Index-Fund Holdings ---\n")

    holdings_script = os.path.join(src_dir, "holdings_snapshots.py")
    run_script(holdings_script, "Build Canonical SEC Holdings Snapshots", env_vars)

    # 2.8. Build direct delisted-security history after holdings so ticker
    # identity can be checked against SEC-observed value/share prices.
    free_history_script = os.path.join(src_dir, "free_price_history.py")
    free_price_file = os.path.join(
        cache_dir, "free_history", "free_history_price_return.parquet"
    )
    if args.refresh_free_history or not os.path.exists(free_price_file):
        refresh_args = ["--refresh"] if args.refresh_free_history else []
        run_script(
            free_history_script,
            "Build Free Historical Delisted Prices",
            env_vars,
            refresh_args,
        )
    else:
        logging.info("--- Reused: Cached Free Historical Delisted Prices ---\n")

    # 3. Reconstruct Weights (scripts/reconstruct_weights.py)
    if not args.skip_reconstruct:
        reconstruct_script = os.path.join(scripts_dir, "reconstruct_weights.py")
        run_script(reconstruct_script, "Reconstruct Index Weights", env_vars)
    else:
        if not os.path.exists(weights_file):
            logging.error(
                f"Cannot use --skip-reconstruct: expected existing weights at {weights_file}, but the file is missing."
            )
            sys.exit(1)
        logging.info(f"--- Skipped: Reconstruct Index Weights (using existing weights at {weights_file}) ---\n")
    
    # 3. Backtest Mega 1.0 total return and the price-return input used by QQUPSIM.
    mega1_script = os.path.join(scripts_dir, "backtest_ndx_mega.py")
    run_ndxmega_backtests(mega1_script, env_vars)
    
    # 4. Backtest Mega 2.0 (scripts/backtest_ndx_mega2.py)
    mega2_script = os.path.join(scripts_dir, "backtest_ndx_mega2.py")
    run_script(mega2_script, "Backtest NDX Mega 2.0", env_vars)

    # 4.5. Backtest NDX30 (scripts/backtest_ndx30.py)
    ndx30_script = os.path.join(scripts_dir, "backtest_ndx30.py")
    run_script(ndx30_script, "Backtest NDX30", env_vars)

    # 5. Validation (scripts/validate_ndx.py)
    validate_script = os.path.join(scripts_dir, "validate_ndx.py")
    run_script(validate_script, "Validate & Compare Results", env_vars)
    
    logging.info("All steps completed successfully.")
    logging.info(
        "Dashboard data (NDXMEGASIM.csv / NDXMEGAPRICESIM.csv / "
        "NDXMEGA2SIM.csv / NDX30SIM.csv) has been updated."
    )
    logging.info(f"Data source used: {data_source_label}")

if __name__ == "__main__":
    main()
