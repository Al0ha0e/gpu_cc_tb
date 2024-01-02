#ifndef OPERATOR_H
#define OPERATOR_H

#ifndef NVRTC_COMPILE

#include <type.cuh>
#include <string>
#include <map>

namespace gputp_operator
{

    class Scope
    {
    public:
        std::map<std::string, common::TypeInfoBase *> names;

        void SetName(std::string name, common::TypeInfoBase *info)
        {
            names[name] = info;
        }

        common::TypeInfoBase *GetTypeByName(std::string name)
        {
            return names[name];
        }

        bool HasName(std::string name)
        {
            auto it = names.find(name);
            return it != names.end();
        }

        void Clear()
        {
            names.clear();
        }
    };

    class Operator
    {
    public:
        Scope *scope;
        common::TypeInfoBase *type;

        Operator() : scope(nullptr), type(nullptr) {}
        Operator(Scope *scp, common::TypeInfoBase *tp) : type(tp), scope(scp) {}
        Operator(Scope *scp) : type(nullptr), scope(scp) {}

        virtual void ToGPUCode(std::string &code) = 0;
        virtual void ApplyScope(Scope *scp) { scope = scp; }
        virtual void TypeCheck() {}
    };

    class IntegerConstantOperator : public Operator
    {
    public:
        long long val;

        IntegerConstantOperator() {}

        IntegerConstantOperator(common::TypeInfoBase *tp, long long v) : Operator(nullptr, tp), val(v) {}

        IntegerConstantOperator(Scope *scp, common::TypeInfoBase *tp, long long v) : Operator(scp, tp), val(v) {}

        void ToGPUCode(std::string &code) override
        {
            code += std::to_string(val);
            ((common::IntegerType *)type)->GetConstPostfix(code);
        }
    };

    class FloatConstantOperator : public Operator
    {
    public:
        double val;

        FloatConstantOperator() {}

        FloatConstantOperator(Scope *scp, common::TypeInfoBase *tp, double v) : Operator(scp, tp), val(v) {}

        void ToGPUCode(std::string &code) override
        {
            code += std::to_string(val);
        }
    };

    class VariableOperator : public Operator
    {
    public:
        std::string name;

        VariableOperator() {}

        VariableOperator(std::string n) : name(n) {}

        VariableOperator(Scope *scp, std::string n) : name(n), Operator(scp, scp->GetTypeByName(n)) {}

        void ToGPUCode(std::string &code) override
        {
            code += name;
        }
    };

    class AssignOperator : public Operator
    {
    public:
        Operator *lval;
        Operator *rval;

        AssignOperator() {}

        AssignOperator(Scope *scp, Operator *l, Operator *r) : Operator(scp), lval(l), rval(r) {}

        void ToGPUCode(std::string &code) override
        {
            lval->ToGPUCode(code);
            code += " = ";
            rval->ToGPUCode(code);
        }

        void ApplyScope(Scope *scp) override
        {
            scope = scp;
            lval->ApplyScope(scp);
            rval->ApplyScope(scp);
        }
    };

    enum BinOperatorType
    {
        ADD,
        MUL
    };

    class BinOperator : public Operator
    {
    public:
        BinOperatorType type;
        Operator *op1;
        Operator *op2;

        BinOperator() {}

        BinOperator(Scope *scp, BinOperatorType tp, Operator *o1, Operator *o2) : Operator(scp), type(tp), op1(o1), op2(o2) {}

        BinOperator(BinOperatorType tp, Operator *o1, Operator *o2) : type(tp), op1(o1), op2(o2) {}

        void ToGPUCode(std::string &code) override
        {
            op1->ToGPUCode(code);
            switch (type)
            {
            case ADD:
                code += " + ";
                break;
            case MUL:
                code += " * ";
            default:
                break;
            }
            op2->ToGPUCode(code);
        }

        void ApplyScope(Scope *scp) override
        {
            scope = scp;
            op1->ApplyScope(scp);
            op2->ApplyScope(scp);
        }
    };

    class ArrayLoadOperator : public Operator
    {
    public:
        Operator *array;
        Operator *offset;

        ArrayLoadOperator() {}

        ArrayLoadOperator(Scope *scp, Operator *arr, Operator *off) : Operator(scp), array(arr), offset(off) {}

        void ToGPUCode(std::string &code) override
        {
            code += "(";
            array->ToGPUCode(code);
            code += ")[";
            offset->ToGPUCode(code);
            code += "]";
        }

        void ApplyScope(Scope *scp) override
        {
            scope = scp;
            array->ApplyScope(scp);
            offset->ApplyScope(scp);
        }
    };

    class StructMemberAccessOperator : public Operator
    {
    public:
        Operator *obj;
        std::string name;

        StructMemberAccessOperator() {}

        StructMemberAccessOperator(Operator *o, std::string n) : obj(o), name(n) {}

        void ToGPUCode(std::string &code) override
        {
            obj->ToGPUCode(code);
            code += ".";
            code += name;
        }

        void ApplyScope(Scope *scp) override
        {
            scope = scp;
            obj->ApplyScope(scp);
        }
    };
}

#endif

#endif