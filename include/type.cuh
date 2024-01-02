#ifndef TYPE_H
#define TYPE_H

#ifndef NVRTC_COMPILE

#include <vector>
#include <string>
#include <sstream>

#endif

namespace common
{
#ifndef NVRTC_COMPILE
    enum TypeCode
    {
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
        STRING
    };

    struct TypeWithSize
    {
        TypeCode type;
        size_t size;
        size_t padding;
        size_t offset;

        TypeWithSize() {}
        TypeWithSize(TypeCode tp, size_t sz, size_t pd) : type(tp), size(sz), padding(pd) {}
    };

    class TypeInfoBase
    {
    public:
        virtual size_t GetSize() = 0;
        virtual std::string GetName() = 0;
    };

    class PlaceHolderType : public TypeInfoBase
    {
        size_t GetSize() override { return 0; }
        std::string GetName() override { return "__placeholder"; }
    };

    class NumericType : public TypeInfoBase
    {
    public:
        virtual size_t GetSize() = 0;
    };

    class IntegerType : public NumericType
    {
    public:
        TypeCode type;
        bool unsign;

        IntegerType() {}

        IntegerType(TypeCode tp, bool unsgn) : type(tp), unsign(unsgn) {}

        size_t GetSize() override
        {
            switch (type)
            {
            case INT8:
                return 1;
            case INT16:
                return 2;
            case INT32:
                return 4;
            default:
                return 8;
            }
        }

        void GetConstPostfix(std::string &code)
        {
            if (unsign)
                code += "U";
            if (type == common::INT64)
                code += "L";
        }

        std::string GetName() override
        {
            switch (type)
            {
            case INT8:
                return unsign ? "unsigned char" : "char";
            case INT16:
                return unsign ? "unsigned short" : "short";
            case INT32:
                return unsign ? "unsigned int" : "int";
            default:
                return unsign ? "unsigned long long" : "long long";
            }
        }
    };

    class FloatType : public NumericType
    {
    public:
        TypeCode type;

        FloatType() {}
        FloatType(TypeCode tp) : type(tp) {}

        size_t GetSize() override
        {
            if (type == FLOAT32)
                return 4;
            return 8;
        }

        std::string GetName() override
        {
            if (type == FLOAT32)
                return "float";
            return "double";
        }
    };

    class StructType : public TypeInfoBase
    {
    public:
        struct StructItem
        {
            std::string name;
            TypeInfoBase *type;
            size_t size;
            size_t padding;
            size_t offset;

            StructItem() {}
            StructItem(std::string n, TypeInfoBase *tp, size_t pd) : name(n), type(tp), size(tp->GetSize()), padding(pd) {}
        };

        std::string name;
        std::vector<StructItem> items;

        StructType() {}

        StructType(std::string n, std::vector<StructItem> &&itms) : name(n), items(itms)
        {
            items[0].offset = 0;
            for (int i = 1; i < items.size(); i++)
                items[i].offset = items[i - 1].offset + items[i - 1].padding + items[i - 1].size;
        }

        size_t GetSize() override
        {
            size_t ret = 0;
            for (auto &item : items)
                ret += item.size + item.padding;
            return ret;
        }

        std::string GetName() override
        {
            std::stringstream ret;

            ret << "struct " << name << "{\n";
            for (auto &item : items)
                ret << item.type->GetName() << " " << item.name << ";\n";
            ret << "};\n";

            return ret.str();
        }
    };

#endif
}

#endif