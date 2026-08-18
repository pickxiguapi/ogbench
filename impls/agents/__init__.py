from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gciql_chunk import GCIQLChunkAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.hiql_chunk import HIQLChunkAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gciql_chunk=GCIQLChunkAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    hiql_chunk=HIQLChunkAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)
