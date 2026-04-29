# Qualitative misclassification samples

NB-BERT correct & 8B 4-shot wrong: 237 / 4340
NB-BERT wrong & 8B 4-shot correct: 259 / 4340
Both wrong: 172 / 4340

## NB-BERT right, 8B 4-shot wrong

- **id=501663** (music, 579 words) true=positive, BERT=positive, LLM raw='negative'
  > Et velsmurt popmaskineri The Weeknd hadde en fin dag på jobben, men jobbet ikke overtid.   The Weeknd har laget noen av de største hitlåtene de siste årene.  Både han og bandet var profesjonelle til fingerspissene denne kvelden, men det var også litt av problemet: det ble akkurat litt for proft, fra begynnelse til slutt.   Bergen Live kan mer enn å lage fest med Elton John, Cecilia Brækhus og Loth...

- **id=300852** (literature, 375 words) true=positive, BERT=positive, LLM raw='negative'
  > Unge norske forfattere skriver heller om samlivsproblemer enn verdensproblemer Denne boka minner oss om at det er noe av det vanskeligste som finnes.   Mange unge norske forfattere beskriver hverdagslige situasjoner, men spriter det opp med noen absurde elementer, så også i Julie T. Stangebyes debutroman.   Jeg-personen i «Lufte gaupene» er en kvinne i starten av tjueåra som har blitt sterkt prege...

- **id=400770** (screen, 599 words) true=positive, BERT=positive, LLM raw='negative'
  > Hvordan kunne svensk politi utpeke Robin (5) og Christian (7) til drapsmenn? Saken Kevin er en rystende dokumentar om hvordan svensk politi presset to brødre på 5 og 7 år til å tilstå et barnedrap.   Sommeren 1998 ble 4-åringen Kevin Hjalmarsson drept i Arvika i Sverige.  Etter en lang og resultatløs etterforskning fokuserte de på to av Kevins naboer; brødrene Robin og Christian, henholdsvis 5 og...

- **id=004769** (screen, 448 words) true=positive, BERT=positive, LLM raw='negative'
  > Flaskepost fra P Fartsfylt jakt på sadistisk seriemorder.   Flaskepost fra P er en fartsfylt spenningskrim som fungerer godt for oss som er glad i en underholdningsfokusert politijakt på sadistisk seriemorder.   Dette er den tredje filmatiseringen av Jussi Adler-Olsens bokhelt Carl Mørck, denne gangen stilsikkert regissert av norske Hans Petter Moland (Aberdeen, Kraftidioten).   Det hele er nydeli...

- **id=300766** (music, 496 words) true=positive, BERT=positive, LLM raw='negative'
  > Som å se pornofilm med sladd The Weeknd har byttet inn sitt hedonistiske særpreg for prinsessa, popstjerne-status og halve bransjeriket.   Mye har skjedd siden forrige gang The Weeknd var på besøk i Oslo.  Den gang som relativt fersk superstjerne i ly av en uvanlig imponerende karriereomveltning.  Platinahoppet fra fiaskoen «Kiss Land» (2013) til «Beauty Behind the Madness» (2015) plasserte ikke b...

## 8B 4-shot right, NB-BERT wrong

- **id=706840** (literature, 373 words) true=negative, BERT=positive, LLM raw='negative'
  > Allianser og intriger fra sagatiden BOK:  Edvard Eikill fortsetter sitt arbeid med å elte sagaene om.   Edvard Eikill:  Malmen og sverdet.  Roman.  222 sider.  Svein Sandnes bokforlag.   Stavangermannen Edvard Eikill (1930) er en sjeldenhet blant forfattere.  Siden han i 1997 pensjonerte seg etter et arbeidsliv som tannlege, har han brukt tida til studier i norrønt språk og litteratur, som i sin t...

- **id=005011** (screen, 541 words) true=negative, BERT=positive, LLM raw='negative'
  > Everything, Everything Flere hakk for glatt, steril og tannløs.   Everything, Everything er en varm og myk drømmefilm med vakre mennesker og vanskelig kjærlighet som får hjerter til å smerte.  Stella Meghie har regissert en historie basert på Nicola Yoons young adult-roman fra 2015.   Dessverre formidles tematikken med en såpeseries florlette estetikk, og sklir etter hvert ut i en retning der trov...

- **id=302665** (screen, 558 words) true=negative, BERT=positive, LLM raw='negative'
  > Et overnaturlig rebusløp uten en eneste skummel scene Sammenlignet med dagens USA er virkeligheten «Rings» presenterer direkte trivelig.   FILM:  En av skrekkfilmens viktigste sosiale funksjoner er å tilby publikum en anledning til å tøye redselsmusklene sine i fellesskap.  I en hverdag fylt av usikkerhet, angst og generell elendighet kan disse musklene bli anspente og stive av krampe, og ved å ti...

- **id=501594** (restaurants, 825 words) true=negative, BERT=positive, LLM raw='negative'
  > Gjennomsnittlige Joe Pregløse sandwicher og supre juicer.  Joe & The Juice serverer sunn hurtigmat i et tregt tempo.   I den gamle strømpebutikken på vei inn i Galleriet har det flyttet inn et nytt konsept.  Danskættede Joe & The Juice lager sunn hurtigmat, til høy musikk og bankende beats.  Her inne er det disko hele dagen, med en hurtigmat-servering bygget som et nattklubbkonsept.  For de tøffes...

- **id=500685** (stage, 671 words) true=positive, BERT=negative, LLM raw='positive'
  > Maurtueliv i en bygård Et litt for tynt materiale å arbeide med.   Bjørn Willberg Andersen kjenner både teatrets indre liv og en viss bygård på hjørnet av Thormøhlens gate og Zetlitz gate på Møhlenpris.  Men det skal mer til for å skape godt teater, også musikkteater.  Når manus sjelden når under overflaten og regien dessuten virker litt gammelmodig og instruktøren ikke har sett hvor det burde kut...

## Both wrong

- **id=301090** (screen, 517 words) true=negative, BERT=positive, LLM raw='positive'
  > Anmeldelse:«The Green Inferno»   Eli Roths groteske kannibalsatire er årets hittil mest anbefalelsesverdige terningkast tre-film.   FILM:  En av de mest beklagelige bieffektene av «found footage»-grøsserens framvekst, er at skrekksjangeren langt på vei har mistet sin satiriske og samfunnskritiske funksjon.Realisme er hittegodsskrekkens fremste mål hva gjelder miljø- og personskildringer, og bestre...

- **id=600268** (screen, 225 words) true=negative, BERT=positive, LLM raw='positive'
  > Haispenning livsfare THRILLER   USA 2016   Regi:  Jaume Collet-Serra Manus:  Anthony Jaswinski   Skuespillere:  Blake Lively, Óscar Jaenada, Sedonna Legge   Musikk:  Marco Beltrami   Aldersgrense:  12 år   «The Shallows» er en film som får mye spenning ut av et nesten beundringsverdig simpelt premiss: blond surferjente blir stuck på skjær med blodtørstig monsterhai sirklende rundt seg.   Til å beg...

- **id=202564** (products, 971 words) true=negative, BERT=positive, LLM raw='positive'
  > Fra håndverker- til familiebil Ford forsøker å lokke til seg hipster-foreldre og ekstremsportutøvere med en barsk utgave av Ranger.   Mote er en ganske forutsigbar greie.  Det er ikke vanskelig å spå når en gammel trend er klar for comeback.  Snakker vi klær, er tidsrammen om lag 30 år.   Bilverdenen bruker mindre tid før motor-moten tar en full 360.  Miljøet har stått i full fokus de siste årene....

- **id=200378** (products, 2388 words) true=positive, BERT=negative, LLM raw='negative'
  > Er vi virkelig klare for å droppe lydutgangen? Apples store endring har ingen åpenbare fordeler.   Nytt år, ny iPhone.  Eller to, da, hvis vi skal være korrekte.  Test av Plus-modellen kommer litt senere i en egen artikkel.   iPhone 7 er den minste av de to nye.  Fremdeles dreier det seg om en 4,7 tommers skjerm og et utseende som, ganske utradisjonelt, ikke har fått de store endringene fra 6s-mod...

- **id=104255** (music, 61 words) true=negative, BERT=positive, LLM raw='positive'
  > 9.Rune Rudberg Band:  «Run Run Away»   Tekst og melodi:  Peter Danielson, Åsa ​Karlström og Mats Larsson   Sympatisk og lettbeint danseband-country i lystig luntetrav mot avslutningens obligatoriske modulering.  Det hadde vært snillisme i praksis å sende denne til Kiev, men den får deg iallfall i godt humør.  Låta er så hyggelig og tilforlatelig at du knapt merker at den har vært der.

