"""Find openly licensed Hugging Face object-detection datasets for untrained classes."""
from huggingface_hub import HfApi
api=HfApi()
terms=["doors","roof detection","cracked tiles","wall materials","bricks","plaster","floor damage"]
allow={"cc-by-4.0","cc0-1.0","apache-2.0","mit","cc-by-sa-4.0"}
seen=set()
for term in terms:
    matches=[]
    for entry in api.list_datasets(search=term,limit=40,full=True):
        if entry.id in seen:continue
        metadata=entry.cardData or {}
        lic=str(metadata.get("license","")).lower()
        task=str(metadata.get("task_categories",""))
        if lic not in allow:continue
        matches.append({"id":entry.id,"license":lic,"task":task,"downloads":entry.downloads})
        seen.add(entry.id)
        if len(matches)>=10:break
    print("SEARCH",term,":",matches,flush=True)
