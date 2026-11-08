"""CLI: python -m forecast1x2 {aggiorna,auto,update-data,predict,backtest,consigli-backtest}"""
import argparse
import sys

from .data import DataError, pipeline_lock


def main(argv=None):
    p = argparse.ArgumentParser(prog="forecast1x2")
    sub = p.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("update-data", help="scarica e normalizza risultati, quote e calendari")
    up.add_argument("--force", action="store_true", help="riscarica anche le stagioni chiuse")
    sub.add_parser("predict", help="previsioni della prossima giornata in outputs/")
    sub.add_parser("backtest", help="calibra gli iperparametri e lancia il backtest walk-forward")
    sub.add_parser("aggiorna", help="update-data + predict: il comando da usare prima di ogni giornata")
    sub.add_parser("auto", help="aggiorna solo se è passato un evento partita (per l'esecuzione pianificata)")
    sub.add_parser("consigli-backtest", help="verifica storica della regola dei consigli in prova (over/under)")
    sub.add_parser("quote-giuste-verifica", help="stima i parametri e verifica sullo storico le quote giuste dei mercati")
    sub.add_parser("quote-planetwin", help="scarica subito le quote Planetwin365 da OddsPapi (serve la chiave)")
    args = p.parse_args(argv)

    if args.cmd == "auto":
        from .auto import run_auto
        return run_auto()
    try:
        with pipeline_lock():
            if args.cmd == "update-data":
                from .data import update_data
                update_data(force=args.force)
            elif args.cmd == "predict":
                from .model import predict_all
                predict_all()
            elif args.cmd == "aggiorna":
                from .data import update_data
                from .model import predict_all
                update_data()
                predict_all()
            elif args.cmd == "backtest":
                from .evaluation import run_backtest
                run_backtest()
            elif args.cmd == "consigli-backtest":
                from .tips import run_rule_backtest
                run_rule_backtest()
            elif args.cmd == "quote-giuste-verifica":
                from .markets import run_calibration
                run_calibration()
            elif args.cmd == "quote-planetwin":
                from .oddspapi import update_bookmaker_odds
                for w in update_bookmaker_odds():
                    print(f"ATTENZIONE: {w}")
    except DataError as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
