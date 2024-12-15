import string

RAG_CONTEXT_INFERENCE = string.Template('$context_pieces\n\nGiven the context pieces above, reply to the following user'
                                        ' query:\n$user_query')
