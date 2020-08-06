
import ATD
ATD.setDefaultKey("paraphrase#thisisit")
errors = ATD.checkDocument("This is are apple")
print(list(errors))
# metrics = ATD.stats("Looking too the water. Fixing your writing typoss.")
# print([str(m) for m in metrics])
#
# for error in errors:
#     print ("%s error for: %s **%s**" % (error.type, error.precontext, error.string))
#     print ("some suggestions: %s" % (", ".join(error.suggestions),))
#
