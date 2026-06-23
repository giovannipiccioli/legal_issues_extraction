"""Classificatore di esistenza dei principi giuridici.

Dato un candidato a "principio giuridico" (stringa estratta da una sentenza),
decide se il principio *esiste davvero* oppure se è stato *inventato* dall'LLM,
e — se esiste — ne restituisce la **forma standard** (canonica).

Metodo: whitelist matching su un elenco di principi *canonici*, dopo una
normalizzazione lessicale (la stessa usata in
``clustering_principi_giuridici.ipynb``). Ogni principio canonico ha una forma
``standard`` e una lista di ``aliases`` (nomi alternativi / sinonimi). Tutte le
superfici (standard + alias) puntano alla stessa forma standard, così varianti
come "buona amministrazione" o "sicurezza giuridica" vengono ricondotte alla
loro forma canonica. La whitelist è interpretata in modo *restrittivo*: si
assume che ogni principio legittimo sia rappresentato lì.

I dati canonici stanno in ``principi_canonici.json`` (generato/aggiornato da
``build_canonici.py`` a partire dalla lista piatta + alias curati a mano).

Priorità di progetto: la pipeline è applicata a uno stream in cui ~5/6 dei
principi sono validi, quindi l'errore costoso è *scartare un principio reale*.
Per questo il matching è volutamente *permissivo* (varianti per superset/subset
dei token), e si massimizza il rifiuto degli inventati solo dopo aver garantito
recall elevatissimo sui validi.

Estensibilità: aggiungere un principio = aggiungere una riga a
``principi_validi_dedup.txt`` (o chiamare ``add_principle``). I parametri
``blocklist`` (forza-invalido) e ``overrides`` (regex forza-valido) permettono
tuning per singolo principio senza toccare il codice.

Dipendenze: solo stdlib. ``rapidfuzz`` è opzionale (``pip install rapidfuzz``):
se assente, il livello fuzzy è semplicemente disabilitato.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
#  Normalizzazione (portata da clustering_principi_giuridici.ipynb)
# --------------------------------------------------------------------------- #

IT_STOPWORDS = {
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
    'di', 'del', 'dei', 'della', 'delle', 'dello', 'degli', 'd',
    'a', 'al', 'allo', 'alla', 'ai', 'agli', 'alle',
    'da', 'dal', 'dallo', 'dalla', 'dai', 'dagli', 'dalle',
    'in', 'nel', 'nello', 'nella', 'nei', 'negli', 'nelle',
    'su', 'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle',
    'con', 'per', 'tra', 'fra', 'e', 'o', 'ed', 'od', 'che', 'non',
    'art', 'artt', 'c', 'cc', 'cpc', 'cpp', 'cost', 'costituzione',
    'codice', 'dlgs', 'dl', 'dpr', 'l', 'lett',
    'ex', 'n', 'nr', 'comma', 'co', 'commi', 'si', 'ne', 'vi', 'ci', 'cui',
    # frammenti di articoli elisi: "dell'IVA" -> "dell" "iva" (l'apostrofo è
    # rimosso dalla punteggiatura, quindi il frammento va tolto qui)
    'dell', 'nell', 'all', 'sull', 'dall', 'quell', 'coll', 'anch',
    # parole di rumore ricorrenti nel dominio
    'principio', 'principi', 'principle', 'principles',
}

PAREN_RE = re.compile(r'\([^)]*\)')
PUNCT_RE = re.compile(r'[^a-z0-9\s]+')
SPACES_RE = re.compile(r'\s+')


def strip_accents(s: str) -> str:
    """Rimuove i segni diacritici (NFKD)."""
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Minuscolo, niente accenti, niente riferimenti normativi fra parentesi,
    niente punteggiatura, spazi compattati."""
    t = strip_accents(text.lower())
    t = PAREN_RE.sub(' ', t)     # via i riferimenti normativi fra parentesi
    t = PUNCT_RE.sub(' ', t)     # via la punteggiatura
    t = SPACES_RE.sub(' ', t).strip()
    return t


def tokens(text: str) -> frozenset:
    """Insieme dei token informativi (>1 carattere, non stopword)."""
    norm = normalize(text)
    return frozenset(w for w in norm.split()
                     if len(w) > 1 and w not in IT_STOPWORDS)


# --------------------------------------------------------------------------- #
#  Classificatore
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Result:
    """Esito della classificazione."""
    exists: bool
    standard_form: str | None  # forma canonica del principio (se riconosciuto)
    reason: str                # quale regola ha deciso (es. 'superset', 'none')
    matched: str | None        # forma whitelist (standard o alias) che ha fatto match

    def __bool__(self) -> bool:  # comodo: ``if classifier.classify(x): ...``
        return self.exists


# tenta l'import di rapidfuzz una sola volta a livello di modulo
try:
    from rapidfuzz import fuzz as _rf_fuzz
except Exception:  # pragma: no cover - dipende dall'ambiente
    _rf_fuzz = None


class PrincipiClassifier:
    """Whitelist matcher per principi giuridici.

    Parametri
    ---------
    principi:
        iterabile di principi. Ogni elemento può essere una stringa (sola forma
        standard) o un dict ``{'standard': ..., 'aliases': [...]}``. Gli alias
        sono nomi alternativi/sinonimi che puntano alla stessa forma standard.
    min_core_tokens:
        un principio valido può fare match per *superset* solo se ha almeno
        questo numero di token. Evita che un valido mono-token (es.
        "Principio di effettività") catturi inventati come "...della notifica".
    enable_subset:
        se True, accetta anche il candidato i cui token sono sottoinsieme di un
        valido (varianti più brevi, es. "Principio di legalità").
    fuzzy_threshold:
        se impostato (0-100) e ``rapidfuzz`` è disponibile, accetta i candidati
        con ``token_set_ratio`` >= soglia. Disattivato di default.
    blocklist:
        stringhe da forzare a *invalido* anche se farebbero match (per
        correggere a mano i pochi falsi positivi noti). Confrontate normalizzate.
    overrides:
        pattern regex (sul testo grezzo, IGNORECASE) da forzare a *valido*.
    """

    def __init__(
        self,
        principi: "Iterable[str | dict]",
        *,
        min_core_tokens: int = 2,
        enable_subset: bool = True,
        fuzzy_threshold: float | None = None,
        blocklist: Iterable[str] | None = None,
        overrides: Iterable[str] | None = None,
    ) -> None:
        self.min_core_tokens = min_core_tokens
        self.enable_subset = enable_subset
        self.fuzzy_threshold = fuzzy_threshold

        # Ogni "superficie" (forma standard o alias) punta a una forma standard.
        # _surfaces: lista di (superficie, norm, tokenset, standard)
        self._surfaces: list[tuple[str, str, frozenset, str]] = []
        self._norm_to_std: dict[str, str] = {}      # norm -> standard
        self._tok_to_std: dict[frozenset, str] = {}  # tokenset -> standard
        self.standards: list[str] = []               # elenco forme standard

        for entry in principi:
            self._add_entry(entry)

        self._blocklist_norm: set[str] = {normalize(b) for b in (blocklist or [])}
        # overrides: regex (sul testo grezzo) che forzano "valido". Possono essere
        # un dict {pattern: forma_standard} oppure un iterabile di pattern (la cui
        # standard_form sarà None).
        self._overrides_std: list[tuple[re.Pattern, str | None]] = []
        if isinstance(overrides, dict):
            items = overrides.items()
        else:
            items = ((p, None) for p in (overrides or []))
        for pat, std in items:
            self._overrides_std.append((re.compile(pat, re.IGNORECASE), std))

    # -- costruzione / estensione ------------------------------------------- #

    def _add_surface(self, surface: str, standard: str) -> None:
        surface = surface.strip()
        if not surface:
            return
        n = normalize(surface)
        tok = tokens(surface)
        self._surfaces.append((surface, n, tok, standard))
        self._norm_to_std.setdefault(n, standard)
        if tok:
            self._tok_to_std.setdefault(tok, standard)

    def _add_entry(self, entry: "str | dict") -> None:
        """Aggiunge un principio. ``entry`` può essere una stringa (solo forma
        standard) oppure un dict ``{'standard': ..., 'aliases': [...]}``."""
        if isinstance(entry, str):
            standard, aliases = entry.strip(), []
        else:
            standard = (entry.get('standard') or '').strip()
            aliases = entry.get('aliases') or []
        if not standard:
            return
        self.standards.append(standard)
        self._add_surface(standard, standard)
        for a in aliases:
            self._add_surface(a, standard)

    def add_principle(self, standard: str, aliases: Iterable[str] = ()) -> None:
        """Aggiunge a runtime una forma standard (con eventuali alias)."""
        self._add_entry({'standard': standard, 'aliases': list(aliases)})

    def add_alias(self, standard: str, alias: str) -> None:
        """Aggancia un alias a una forma standard già presente."""
        self._add_surface(alias, standard)

    @classmethod
    def from_files(cls, *paths: str | Path, **kwargs) -> "PrincipiClassifier":
        """Costruisce dal formato piatto (un principio per riga). Ogni riga è
        una forma standard senza alias. Mantenuto per retro-compatibilità."""
        principi: list[str] = []
        for path in paths:
            with open(path, encoding='utf-8') as fh:
                principi.extend(line.strip() for line in fh if line.strip())
        return cls(principi, **kwargs)

    @classmethod
    def from_canonical_file(cls, path: str | Path,
                            **kwargs) -> "PrincipiClassifier":
        """Costruisce dal formato canonico JSON:
        ``[{"standard": ..., "aliases": [...]}, ...]``."""
        with open(path, encoding='utf-8') as fh:
            entries = json.load(fh)
        return cls(entries, **kwargs)

    # -- classificazione ----------------------------------------------------- #

    def classify(self, text: str) -> Result:
        n = normalize(text)
        tok = tokens(text)

        # 1) blocklist: forza invalido
        if n in self._blocklist_norm:
            return Result(False, None, 'blocklist', None)

        # 2) overrides: forza valido (pattern aggiunti a mano)
        for pat, std in self._overrides_std:
            if pat.search(text):
                return Result(True, std, 'override', pat.pattern)

        # 3) match esatto sulla forma normalizzata
        std = self._norm_to_std.get(n)
        if std is not None:
            return Result(True, std, 'exact-norm', text)

        # 4) match esatto sull'insieme dei token
        if tok:
            std = self._tok_to_std.get(tok)
            if std is not None:
                return Result(True, std, 'tokset', None)

        # 5) superset: i token di una superficie (>= min_core_tokens) ⊆ candidato
        #    -> il candidato è una variante più specifica di un principio reale
        for surface, _, vtok, std in self._surfaces:
            if len(vtok) >= self.min_core_tokens and vtok <= tok:
                return Result(True, std, 'superset', surface)

        # 6) subset: i token del candidato (>= min_core_tokens) ⊆ una superficie
        #    -> il candidato è una variante più breve di un principio reale
        if self.enable_subset and len(tok) >= self.min_core_tokens:
            for surface, _, vtok, std in self._surfaces:
                if tok <= vtok:
                    return Result(True, std, 'subset', surface)

        # 7) fuzzy opzionale (rapidfuzz)
        if self.fuzzy_threshold is not None and _rf_fuzz is not None and n:
            best, best_score = None, 0.0
            for surface, vn, _, std in self._surfaces:
                s = _rf_fuzz.token_set_ratio(n, vn)
                if s > best_score:
                    best, best_score = (surface, std), s
            if best is not None and best_score >= self.fuzzy_threshold:
                return Result(True, best[1], f'fuzzy:{best_score:.0f}', best[0])

        return Result(False, None, 'none', None)

    def exists(self, text: str) -> bool:
        """True se il principio è ritenuto esistente."""
        return self.classify(text).exists

    def standard_form(self, text: str) -> str | None:
        """Forma canonica del principio, o None se non riconosciuto."""
        return self.classify(text).standard_form


# --------------------------------------------------------------------------- #
#  Istanza di default (whitelist = principi_validi_dedup.txt)
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve().parent
DEFAULT_CANONICAL_PATH = _HERE / 'principi_canonici.json'   # sorgente canonica
DEFAULT_VALID_PATH = _HERE / 'principi_validi_dedup.txt'    # fallback piatto

# Falsi positivi sottili noti (varianti subset/superset etichettate come
# inventate nel ground truth). Lasciati VUOTI di default per privilegiare il
# recall sui validi; popolare per guadagnare precisione a scapito del recall.
DEFAULT_BLOCKLIST: list[str] = []

_default_classifier: PrincipiClassifier | None = None


def get_default_classifier() -> PrincipiClassifier:
    """Restituisce (creandolo una volta sola) il classificatore di default.
    Usa ``principi_canonici.json`` se presente, altrimenti la lista piatta."""
    global _default_classifier
    if _default_classifier is None:
        if DEFAULT_CANONICAL_PATH.exists():
            _default_classifier = PrincipiClassifier.from_canonical_file(
                DEFAULT_CANONICAL_PATH, blocklist=DEFAULT_BLOCKLIST)
        else:
            _default_classifier = PrincipiClassifier.from_files(
                DEFAULT_VALID_PATH, blocklist=DEFAULT_BLOCKLIST)
    return _default_classifier


def exists(text: str) -> bool:
    """Scorciatoia: usa il classificatore di default."""
    return get_default_classifier().exists(text)


def classify(text: str) -> Result:
    """Scorciatoia: usa il classificatore di default."""
    return get_default_classifier().classify(text)


def standard_form(text: str) -> str | None:
    """Scorciatoia: forma canonica secondo il classificatore di default."""
    return get_default_classifier().standard_form(text)


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        r = classify(arg)
        flag = 'ESISTE' if r.exists else 'INVENTATO'
        line = f'{flag:10s} [{r.reason}] {arg!r}'
        if r.standard_form and normalize(r.standard_form) != normalize(arg):
            line += f'  ->  {r.standard_form!r}'
        print(line)
