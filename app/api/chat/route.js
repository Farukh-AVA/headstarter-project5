import { NextResponse } from "next/server";
import { Pinecone } from '@pinecone-database/pinecone'; 
import {OpenAI} from "openai";

const systemPrompt = 
`
You are a helpful and knowledgeable assistant designed to help students find professors according to their specific queries. When a student asks for information about professors, you will use Retrieval-Augmented Generation (RAG) to retrieve relevant data and provide information on three professors that match their query.

For each user query, you should:

Understand the student's requirements based on their query.
Retrieve and present information on three professors that best match the query. This includes their names, the subjects they teach, their ratings, and a brief review.
Ensure the information is accurate, concise, and easy to understand.
If the student's query is too broad, ask for clarification to narrow down the search.
Always be polite and aim to provide the most relevant and useful information to the student.
You may also include suggestions if the student asks for additional recommendations or further details about the professors provided.

`

export async function POST(req) {
    
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
    })

    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text, 
        encoding_format: 'float'
    })

    const results = await await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = `\n\n Returned results from vector db (done automatically)`

    results.matches.forEach((match) => {
        
        resultString += ` \n
            Professor: ${match.id},
            Review: ${match.metadata.sstars},
            Subject: ${match.subject},
            Stars: ${match.metadata.sstars},
            \n\n
        `
    });

    const lastMessage = data[data.length-1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length -1)

    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt}, 
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent}
        ], 
        model: 'gpt-3.5-turbo', 
        stream: true, 
    })

    const stream = new ReadableStream({
        async start(controlller){
            const encoder = new TextEncoder()
            try{
                for await (const chunk of completion){
                    const content = chunk.choices[0]?.delta.content
                    if(content){
                        const text = encoder.encode(content)
                        controlller.enqueue(text)
                    }
                }
            }catch(err){
                controlller.error(err)
            }finally{
                controlller.close()
            }
        }
    })
    return new NextResponse(stream)
}